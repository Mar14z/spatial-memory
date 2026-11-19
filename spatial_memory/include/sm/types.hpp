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
};

struct Config {
    float grid_size = 1.0f;
    float assoc_radius = 0.5f;
    int64_t assoc_window_ns = 5'000'000'000LL;
    uint32_t max_entities = 1'000'000;
    std::string db_path;  // empty = in-memory only
};

}  // namespace sm
