#include "sm/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <unordered_set>
#include <utility>

namespace sm {

namespace {
constexpr size_t kFlushBatch = 100;
constexpr auto kFlushInterval = std::chrono::milliseconds(50);
}  // namespace

SpatialMemoryEngine::SpatialMemoryEngine(Config cfg)
    : cfg_(std::move(cfg)), grid_(cfg_.grid_size) {
    if (!cfg_.db_path.empty()) {
        store_ = std::make_unique<SqliteStore>();
        if (store_->open(cfg_.db_path)) {
            recover();
            flush_thread_ = std::thread([this] { flush_loop(); });
        } else {
            fprintf(stderr, "[engine] persistence disabled: open failed\n");
            store_.reset();
        }
    }
}

SpatialMemoryEngine::~SpatialMemoryEngine() {
    stop_ = true;
    q_cv_.notify_all();
    if (flush_thread_.joinable()) flush_thread_.join();

    std::deque<WriteItem> rest;
    {
        std::lock_guard<std::mutex> lk(q_mtx_);
        rest.swap(queue_);
    }
    flush_batch(rest);
    if (store_) store_->close();
}

QueryResult SpatialMemoryEngine::to_result(const Entity& e) {
    QueryResult r;
    r.id = e.id;
    r.label = e.label;
    r.pos[0] = e.pos[0];
    r.pos[1] = e.pos[1];
    r.pos[2] = e.pos[2];
    r.size[0] = e.size[0];
    r.size[1] = e.size[1];
    r.size[2] = e.size[2];
    r.last_seen_ns = e.last_seen_ns;
    r.confidence = e.confidence;
    r.state = static_cast<uint8_t>(e.state);
    return r;
}

void SpatialMemoryEngine::recover() {
    store_->load_entities([&](const Entity& e) {
        if (e.state == EntityState::kDeleted) return;
        uint32_t slot = pool_.alloc();
        Entity& slot_e = pool_.get(slot);
        slot_e = e;
        grid_.insert(slot, slot_e.pos);
        id_to_slot_[e.id] = slot;
        by_label_[e.label].push_back(slot);
        slot_e.label_pos = (uint32_t)(by_label_[e.label].size() - 1);
        if (e.id >= next_id_) next_id_ = e.id + 1;
    });
    // Deleted rows are skipped above, so derive the id watermark from the table itself.
    uint64_t max_id = store_->max_entity_id();
    if (max_id >= next_id_) next_id_ = max_id + 1;
    store_->load_features([&](uint64_t id, const float* f, size_t n) {
        if (n == 0) return;
        // Drop orphan features whose entity row is gone (crash between two batch steps).
        auto sit = id_to_slot_.find(id);
        if (sit == id_to_slot_.end()) return;
        if (feat_dim_ == 0) feat_dim_ = n;
        if (n != feat_dim_) return;
        feat_row_[id] = (uint32_t)feat_ids_.size();
        feat_ids_.push_back(id);
        feat_flat_.insert(feat_flat_.end(), f, f + n);
    });
}

void SpatialMemoryEngine::enqueue(const Entity& e, const Observation& o) {
    WriteItem wi;
    wi.kind = WriteItem::kObservation;
    wi.ent = e;
    wi.obs = o;
    enqueue_item(std::move(wi));
}

void SpatialMemoryEngine::enqueue_item(WriteItem&& wi) {
    // Memory-only engine has no consumer for the queue: never accumulate writes.
    if (!store_) return;
    bool notify = false;
    {
        std::lock_guard<std::mutex> lk(q_mtx_);
        queue_.push_back(std::move(wi));
        notify = queue_.size() >= kFlushBatch;
    }
    if (notify) q_cv_.notify_one();
}

void SpatialMemoryEngine::flush_loop() {
    std::deque<WriteItem> batch;
    while (!stop_.load()) {
        {
            std::unique_lock<std::mutex> lk(q_mtx_);
            q_cv_.wait_for(lk, kFlushInterval, [this] {
                return stop_.load() || queue_.size() >= kFlushBatch;
            });
            batch.swap(queue_);
        }
        if (!batch.empty()) {
            flush_batch(batch);
            batch.clear();
        }
    }
}

void SpatialMemoryEngine::flush_batch(std::deque<WriteItem>& batch) {
    if (!store_ || batch.empty()) return;
    std::lock_guard<std::mutex> lk(store_mtx_);
    store_->begin();
    // Entity row deletions must run last, otherwise a later upsert would resurrect them.
    std::unordered_set<uint64_t> dels;
    for (auto& it : batch) {
        if (it.kind == WriteItem::kDelete) dels.insert(it.id);
    }
    // 1) entity rows first (features reference them). Forgets are upserted as tombstones
    //    so the row exists even when the entity was never flushed alive.
    std::unordered_map<uint64_t, const Entity*> last;
    last.reserve(batch.size());
    for (auto& it : batch) {
        if (it.kind == WriteItem::kObservation) last[it.ent.id] = &it.ent;
        else if (it.kind == WriteItem::kDelete) last[it.id] = &it.ent;
    }
    for (auto& kv : last) store_->upsert_entity(*kv.second);
    // 2) timeline segments + feature blobs (timeline is history, kept even after forget)
    for (auto& it : batch) {
        if (it.kind == WriteItem::kObservation) {
            store_->append_segment(it.ent.id, it.obs,
                                   cfg_.segment_max_gap_ns, cfg_.segment_max_dist_m);
        } else if (it.kind == WriteItem::kFeature) {
            store_->set_feature(it.id, it.feat.data(), it.feat.size());
        }
    }
    // 3) forget: clear the feature blob and stamp the tombstone
    for (uint64_t id : dels) store_->delete_entity(id);
    store_->commit();
    persisted_ += batch.size();
}

uint64_t SpatialMemoryEngine::add_observation(const Observation& obs) {
    stats_.adds++;

    float best_d2 = cfg_.assoc_radius * cfg_.assoc_radius;
    uint32_t best_slot = UINT32_MAX;

    grid_.for_candidates(obs.pos, cfg_.assoc_radius, [&](uint32_t slot) {
        const Entity& e = pool_.get(slot);
        if (e.state == EntityState::kDeleted) return;
        if (e.label != obs.label) return;
        if (obs.ts_ns - e.last_seen_ns > cfg_.assoc_window_ns) return;
        float dx = e.pos[0] - obs.pos[0];
        float dy = e.pos[1] - obs.pos[1];
        float dz = e.pos[2] - obs.pos[2];
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 <= best_d2) {
            best_d2 = d2;
            best_slot = slot;
        }
    });

    if (best_slot != UINT32_MAX) {
        Entity& e = pool_.get(best_slot);
        grid_.remove(best_slot, e.pos);
        e.pos[0] = obs.pos[0];
        e.pos[1] = obs.pos[1];
        e.pos[2] = obs.pos[2];
        grid_.insert(best_slot, e.pos);
        e.size[0] = obs.size[0];
        e.size[1] = obs.size[1];
        e.size[2] = obs.size[2];
        e.last_seen_ns = obs.ts_ns;
        e.obs_count++;
        e.confidence += (obs.confidence - e.confidence) * 0.2f;
        e.state = EntityState::kActive;  // re-observed: teleport stale back to active
        stats_.updates++;
        const uint64_t id = e.id;
        enqueue(e, obs);
        if ((++ops_since_sweep_ & 1023u) == 0) sweep(obs.ts_ns);
        return id;
    }

    uint32_t slot = pool_.alloc();
    Entity& e = pool_.get(slot);
    e.id = next_id_++;
    e.label = obs.label;
    e.state = EntityState::kActive;
    e.pos[0] = obs.pos[0];
    e.pos[1] = obs.pos[1];
    e.pos[2] = obs.pos[2];
    e.size[0] = obs.size[0];
    e.size[1] = obs.size[1];
    e.size[2] = obs.size[2];
    e.first_seen_ns = obs.ts_ns;
    e.last_seen_ns = obs.ts_ns;
    e.confidence = obs.confidence;
    e.obs_count = 1;

    grid_.insert(slot, e.pos);
    id_to_slot_[e.id] = slot;
    by_label_[e.label].push_back(slot);
    e.label_pos = (uint32_t)(by_label_[e.label].size() - 1);
    stats_.new_entities++;
    const uint64_t id = e.id;
    enqueue(e, obs);
    if ((++ops_since_sweep_ & 1023u) == 0) sweep(obs.ts_ns);
    return id;
}

std::vector<QueryResult> SpatialMemoryEngine::query_radius(
    const float center[3], float radius, std::optional<uint16_t> label) const {
    stats_.queries++;
    std::vector<QueryResult> out;
    const float r2 = radius * radius;
    grid_.for_candidates(center, radius, [&](uint32_t slot) {
        const Entity& e = pool_.get(slot);
        if (e.state == EntityState::kDeleted) return;
        if (label && e.label != *label) return;
        float dx = e.pos[0] - center[0];
        float dy = e.pos[1] - center[1];
        float dz = e.pos[2] - center[2];
        if (dx * dx + dy * dy + dz * dz <= r2) out.push_back(to_result(e));
    });
    return out;
}

std::vector<QueryResult> SpatialMemoryEngine::query_label(uint16_t label) const {
    stats_.queries++;
    std::vector<QueryResult> out;
    auto it = by_label_.find(label);
    if (it == by_label_.end()) return out;
    out.reserve(it->second.size());
    for (uint32_t slot : it->second) {
        const Entity& e = pool_.get(slot);
        if (e.state == EntityState::kDeleted) continue;
        out.push_back(to_result(e));
    }
    return out;
}

std::optional<QueryResult> SpatialMemoryEngine::get(uint64_t id) const {
    stats_.queries++;
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) return std::nullopt;
    const Entity& e = pool_.get(it->second);
    if (e.state == EntityState::kDeleted) return std::nullopt;
    return to_result(e);
}

SpatialMemoryEngine::Stats SpatialMemoryEngine::stats() const {
    Stats s = stats_;
    s.persisted_items = persisted_.load();
    return s;
}

uint64_t SpatialMemoryEngine::persisted_observations() {
    if (!store_) return 0;
    std::lock_guard<std::mutex> lk(store_mtx_);
    return store_->count_observations();
}

uint64_t SpatialMemoryEngine::persisted_segments() {
    if (!store_) return 0;
    std::lock_guard<std::mutex> lk(store_mtx_);
    return store_->count_segments();
}

std::vector<HistoryPoint> SpatialMemoryEngine::query_history(
    uint64_t entity_id, int64_t from_ts, int64_t to_ts, size_t limit) const {
    std::vector<HistoryPoint> out;
    if (!store_) return out;
    std::lock_guard<std::mutex> lk(store_mtx_);
    store_->query_history(entity_id, from_ts, to_ts, limit,
                          [&](const HistoryPoint& hp) { out.push_back(hp); });
    return out;
}

bool SpatialMemoryEngine::set_feature(uint64_t id, const float* feat, size_t dim) {
    if (!feat || dim == 0) return false;
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) return false;
    if (pool_.get(it->second).state == EntityState::kDeleted) return false;

    std::vector<float> v(feat, feat + dim);
    double s = 0.0;
    for (float x : v) s += (double)x * (double)x;
    if (s <= 0.0) return false;
    float inv = (float)(1.0 / std::sqrt(s));
    for (float& x : v) x *= inv;

    auto row_it = feat_row_.find(id);
    if (row_it != feat_row_.end()) {
        if (dim != feat_dim_) return false;
        std::copy(v.begin(), v.end(), feat_flat_.begin() + (size_t)row_it->second * feat_dim_);
    } else {
        if (feat_dim_ == 0) feat_dim_ = dim;
        if (dim != feat_dim_) return false;
        feat_row_[id] = (uint32_t)feat_ids_.size();
        feat_ids_.push_back(id);
        feat_flat_.insert(feat_flat_.end(), v.begin(), v.end());
    }

    if (store_) {
        WriteItem wi;
        wi.kind = WriteItem::kFeature;
        wi.id = id;
        wi.feat = std::move(v);
        enqueue_item(std::move(wi));
    }
    return true;
}

std::vector<SpatialMemoryEngine::FeatureHit> SpatialMemoryEngine::query_by_feature(
    const float* feat, size_t dim, size_t top_k) const {
    std::vector<FeatureHit> out;
    if (!feat || dim == 0 || feat_ids_.empty() || top_k == 0) return out;
    if (dim != feat_dim_) return out;

    std::vector<float> q(feat, feat + dim);
    double s = 0.0;
    for (float x : q) s += (double)x * (double)x;
    if (s <= 0.0) return out;
    float inv = (float)(1.0 / std::sqrt((float)s));
    for (float& x : q) x *= inv;

    const size_t n = feat_ids_.size();
    std::vector<std::pair<float, uint32_t>> cands(n);
    const float* base = feat_flat_.data();
    for (size_t r = 0; r < n; r++) {
        const float* f = base + r * dim;
        float dot = 0.f;
        for (size_t i = 0; i < dim; i++) dot += q[i] * f[i];
        cands[r] = {dot, (uint32_t)r};
    }

    size_t k = std::min(top_k, n);
    std::partial_sort(cands.begin(), cands.begin() + (ptrdiff_t)k, cands.end(),
                      [](const std::pair<float, uint32_t>& a,
                         const std::pair<float, uint32_t>& b) { return a.first > b.first; });

    out.reserve(k);
    for (size_t i = 0; i < k; i++) {
        FeatureHit h;
        h.id = feat_ids_[cands[i].second];
        h.score = cands[i].first;
        auto it = id_to_slot_.find(h.id);
        if (it != id_to_slot_.end()) {
            const Entity& e = pool_.get(it->second);
            h.label = e.label;
            h.pos[0] = e.pos[0];
            h.pos[1] = e.pos[1];
            h.pos[2] = e.pos[2];
            h.last_seen_ns = e.last_seen_ns;
        }
        out.push_back(h);
    }
    return out;
}

std::vector<HistoryPoint> SpatialMemoryEngine::query_time_range(
    int64_t from_ts, int64_t to_ts, std::optional<uint16_t> label, size_t limit) const {
    std::vector<HistoryPoint> out;
    if (!store_) return out;
    std::lock_guard<std::mutex> lk(store_mtx_);
    store_->query_time_range(from_ts, to_ts, label ? (int)*label : -1, limit,
                             [&](const HistoryPoint& hp) { out.push_back(hp); });
    return out;
}

void SpatialMemoryEngine::erase_feature_row(uint64_t id) {
    auto it = feat_row_.find(id);
    if (it == feat_row_.end() || feat_dim_ == 0) return;
    const uint32_t row = it->second;
    const size_t last = feat_ids_.size() - 1;
    if (row != last) {
        // Move the last row into the hole, then fix up its reverse mapping.
        float* dst = feat_flat_.data() + (size_t)row * feat_dim_;
        const float* src = feat_flat_.data() + (size_t)last * feat_dim_;
        std::copy(src, src + feat_dim_, dst);
        const uint64_t moved = feat_ids_[last];
        feat_ids_[row] = moved;
        feat_row_[moved] = row;
    }
    feat_ids_.pop_back();
    feat_flat_.resize(last * feat_dim_);
    feat_row_.erase(id);
}

void SpatialMemoryEngine::forget_entity(uint32_t slot) {
    if (slot >= pool_.capacity()) return;
    Entity& e = pool_.get(slot);
    if (e.state == EntityState::kDeleted) return;
    const uint64_t id = e.id;
    const uint16_t label = e.label;

    grid_.remove(slot, e.pos);
    auto lit = by_label_.find(label);
    if (lit != by_label_.end()) {
        auto& v = lit->second;
        const uint32_t pos = e.label_pos;
        if (pos < v.size() && v[pos] == slot) {
            const uint32_t moved = v.back();
            v[pos] = moved;
            if (moved != slot) pool_.get(moved).label_pos = pos;
            v.pop_back();
        } else {
            for (size_t i = 0; i < v.size(); ++i) {  // defensive: index drift must not leak slots
                if (v[i] != slot) continue;
                const uint32_t moved = v.back();
                v[i] = moved;
                if (moved != slot) pool_.get(moved).label_pos = (uint32_t)i;
                v.pop_back();
                break;
            }
        }
        if (v.empty()) by_label_.erase(lit);
    }
    id_to_slot_.erase(id);
    erase_feature_row(id);

    if (store_) {
        WriteItem wi;
        wi.kind = WriteItem::kDelete;
        wi.id = id;
        wi.ent = e;
        wi.ent.state = EntityState::kDeleted;  // tombstone payload for the persistent row
        enqueue_item(std::move(wi));
    }
    pool_.free_slot(slot);
    stats_.forgotten++;
}

void SpatialMemoryEngine::enforce_capacity() {
    if (pool_.live_count() <= cfg_.max_entities) return;
    std::vector<std::pair<int64_t, uint32_t>> all;
    all.reserve(pool_.live_count());
    for (uint32_t slot = 0; slot < pool_.capacity(); ++slot) {
        const Entity& e = pool_.get(slot);
        if (e.state == EntityState::kDeleted) continue;
        all.emplace_back(e.last_seen_ns, slot);
    }
    // Drop the oldest 10% over the limit, so eviction does not run on every observation.
    const size_t target = (size_t)cfg_.max_entities * 9 / 10;
    if (all.size() <= target) return;
    const size_t drop = all.size() - target;
    std::nth_element(all.begin(), all.begin() + (ptrdiff_t)drop, all.end());
    for (size_t i = 0; i < drop; ++i) forget_entity(all[i].second);
}

void SpatialMemoryEngine::sweep(int64_t now_ns) {
    const size_t cap = pool_.capacity();
    for (uint32_t slot = 0; slot < cap; ++slot) {
        Entity& e = pool_.get(slot);
        if (e.state == EntityState::kDeleted) continue;
        const int64_t age = now_ns - e.last_seen_ns;
        if (age < 0) continue;  // out-of-order timestamps must not evict fresh data
        if (age > cfg_.forget_after_ns) {
            forget_entity(slot);
        } else if (age > cfg_.stale_after_ns) {
            e.state = EntityState::kStale;
        }
    }
    enforce_capacity();
}

bool SpatialMemoryEngine::remove_entity(uint64_t id) {
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) return false;
    forget_entity(it->second);
    return true;
}

size_t SpatialMemoryEngine::pending_writes() const {
    std::lock_guard<std::mutex> lk(q_mtx_);
    return queue_.size();
}

}  // namespace sm
