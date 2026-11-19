#pragma once
#include <optional>
#include <unordered_map>
#include <vector>

#include "sm/entity_pool.hpp"
#include "sm/grid_index.hpp"
#include "sm/types.hpp"

namespace sm {

class SpatialMemoryEngine {
public:
    explicit SpatialMemoryEngine(Config cfg = {});

    // Ingest one detection. Returns stable entity id.
    uint64_t add_observation(const Observation& obs);

    std::vector<QueryResult> query_radius(const float center[3], float radius,
                                          std::optional<uint16_t> label = std::nullopt) const;
    std::vector<QueryResult> query_label(uint16_t label) const;
    std::optional<QueryResult> get(uint64_t id) const;

    struct Stats {
        uint64_t adds = 0;
        uint64_t updates = 0;
        uint64_t new_entities = 0;
        uint64_t queries = 0;
    };

    Stats stats() const { return stats_; }
    size_t size() const { return pool_.live_count(); }
    size_t grid_cells() const { return grid_.cell_count(); }
    const Entity& entity(uint32_t slot) const { return pool_.get(slot); }

private:
    static QueryResult to_result(const Entity& e);

    Config cfg_;
    EntityPool pool_;
    GridIndex grid_;
    std::unordered_map<uint64_t, uint32_t> id_to_slot_;
    std::unordered_map<uint16_t, std::vector<uint32_t>> by_label_;
    uint64_t next_id_ = 1;
    mutable Stats stats_{};
};

}  // namespace sm
