#pragma once
#include <cstdint>
#include <string>

namespace sm {

struct Observation {
    uint16_t label = 0;
    float pos[3]  = {0.f, 0.f, 0.f};
    float size[3] = {0.1f, 0.1f, 0.1f};
    int64_t ts_ns = 0;
    float confidence = 0.f;
};

struct QueryResult {
    uint64_t id = 0;
    uint16_t label = 0;
    float pos[3]  = {0.f, 0.f, 0.f};
    float size[3] = {0.f, 0.f, 0.f};
    int64_t last_seen_ns = 0;
    float confidence = 0.f;
    uint8_t state = 0;  // EntityState: 0=active, 1=stale (still queryable)
};

enum class EntityState : uint8_t { kActive = 0, kStale = 1, kDeleted = 2 };

struct Entity {
    uint64_t id = 0;
    uint16_t label = 0;
    EntityState state = EntityState::kActive;
    float pos[3]  = {0.f, 0.f, 0.f};
    float size[3] = {0.f, 0.f, 0.f};
    int64_t first_seen_ns = 0;
    int64_t last_seen_ns = 0;
    float confidence = 0.f;
    uint32_t obs_count = 0;
    uint32_t label_pos = 0;  // slot's index inside by_label_[label], for O(1) removal
};

struct Config {
    float grid_size = 1.0f;
    float assoc_radius = 0.5f;
    int64_t assoc_window_ns = 5'000'000'000LL;
    uint32_t max_entities = 1'000'000;
    std::string db_path;  // empty = in-memory only

    // v2: observation compression (resident segments)
    int64_t segment_max_gap_ns = 5'000'000'000LL;  // 5s: merge observations closer than this
    float segment_max_dist_m = 0.05f;              // 5cm: and spatially closer than this

    // v3d: entity lifecycle. Stale entities stay queryable (memory is meant to be kept);
    // forgotten entities are removed from indexes, feature table and SQLite.
    int64_t stale_after_ns = 120'000'000'000LL;     // 2min without observation -> kStale
    int64_t forget_after_ns = 3'600'000'000'000LL;  // 1h -> delete
};

}  // namespace sm
