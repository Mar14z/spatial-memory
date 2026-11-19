#include "sm/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
    return r;
}

void SpatialMemoryEngine::recover() {
    store_->load_entities([&](const Entity& e) {
        uint32_t slot = pool_.alloc();
        pool_.get(slot) = e;
        grid_.insert(slot, e.pos);
        id_to_slot_[e.id] = slot;
        by_label_[e.label].push_back(slot);
        if (e.id >= next_id_) next_id_ = e.id + 1;
    });
    store_->load_features([&](uint64_t id, const float* f, size_t n) {
        if (n == 0) return;
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
    // 1) entity rows first (features reference them)
    std::unordered_map<uint64_t, const Entity*> last;
    last.reserve(batch.size());
    for (auto& it : batch) {
        if (it.kind == WriteItem::kObservation) last[it.ent.id] = &it.ent;
    }
    for (auto& kv : last) store_->upsert_entity(*kv.second);
    // 2) timeline segments + feature blobs
    for (auto& it : batch) {
        if (it.kind == WriteItem::kObservation) {
            store_->append_segment(it.ent.id, it.obs,
                                   cfg_.segment_max_gap_ns, cfg_.segment_max_dist_m);
        } else {
            store_->set_feature(it.id, it.feat.data(), it.feat.size());
        }
    }
    store_->commit();
    persisted_ += batch.size();
}

uint64_t SpatialMemoryEngine::add_observation(const Observation& obs) {
    stats_.adds++;

    float best_d2 = cfg_.assoc_radius * cfg_.assoc_radius;
    uint32_t best_slot = UINT32_MAX;

    grid_.for_candidates(obs.pos, cfg_.assoc_radius, [&](uint32_t slot) {
        const Entity& e = pool_.get(slot);
        if (e.state != EntityState::kActive) return;
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
        stats_.updates++;
        enqueue(e, obs);
        return e.id;
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
    stats_.new_entities++;
    enqueue(e, obs);
    return e.id;
}

std::vector<QueryResult> SpatialMemoryEngine::query_radius(
    const float center[3], float radius, std::optional<uint16_t> label) const {
    stats_.queries++;
    std::vector<QueryResult> out;
    const float r2 = radius * radius;
    grid_.for_candidates(center, radius, [&](uint32_t slot) {
        const Entity& e = pool_.get(slot);
        if (e.state != EntityState::kActive) return;
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
        if (e.state != EntityState::kActive) continue;
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

}  // namespace sm
