#include "sm/engine.hpp"

#include <cstdint>

namespace sm {

SpatialMemoryEngine::SpatialMemoryEngine(Config cfg)
    : cfg_(cfg), grid_(cfg.grid_size) {}

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

}  // namespace sm
