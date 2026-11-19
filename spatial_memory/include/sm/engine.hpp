#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include "sm/entity_pool.hpp"
#include "sm/grid_index.hpp"
#include "sm/sqlite_store.hpp"
#include "sm/types.hpp"

namespace sm {

class SpatialMemoryEngine {
public:
    explicit SpatialMemoryEngine(Config cfg = {});
    ~SpatialMemoryEngine();
    SpatialMemoryEngine(const SpatialMemoryEngine&) = delete;
    SpatialMemoryEngine& operator=(const SpatialMemoryEngine&) = delete;

    // Ingest one detection. Returns stable entity id.
    uint64_t add_observation(const Observation& obs);

    std::vector<QueryResult> query_radius(const float center[3], float radius,
                                          std::optional<uint16_t> label = std::nullopt) const;
    std::vector<QueryResult> query_label(uint16_t label) const;
    std::optional<QueryResult> get(uint64_t id) const;

    // v2: timeline queries (persistent mode only; empty result in memory mode)
    std::vector<HistoryPoint> query_history(uint64_t entity_id, int64_t from_ts, int64_t to_ts,
                                            size_t limit = 1000) const;
    std::vector<HistoryPoint> query_time_range(int64_t from_ts, int64_t to_ts,
                                               std::optional<uint16_t> label = std::nullopt,
                                               size_t limit = 10000) const;

    // v3: semantic features (e.g. CLIP 512-d). Stored L2-normalized; cosine == dot.
    struct FeatureHit {
        uint64_t id = 0;
        float score = 0.f;
        uint16_t label = 0;
        float pos[3] = {0.f, 0.f, 0.f};
        int64_t last_seen_ns = 0;
    };
    bool set_feature(uint64_t id, const float* feat, size_t dim);
    std::vector<FeatureHit> query_by_feature(const float* feat, size_t dim,
                                             size_t top_k = 5) const;
    size_t feature_count() const { return feat_ids_.size(); }

    struct Stats {
        uint64_t adds = 0;
        uint64_t updates = 0;
        uint64_t new_entities = 0;
        uint64_t queries = 0;
        uint64_t persisted_items = 0;
    };

    Stats stats() const;
    size_t size() const { return pool_.live_count(); }
    size_t grid_cells() const { return grid_.cell_count(); }
    bool persistent() const { return store_ != nullptr; }
    uint64_t persisted_observations();
    uint64_t persisted_segments();

private:
    struct WriteItem {
        enum Kind : uint8_t { kObservation = 0, kFeature = 1 };
        Kind kind = kObservation;
        Entity ent;               // kObservation
        Observation obs;          // kObservation
        uint64_t id = 0;          // kFeature
        std::vector<float> feat;  // kFeature (L2-normalized)
    };

    void recover();
    void enqueue(const Entity& e, const Observation& o);
    void enqueue_item(WriteItem&& wi);
    void flush_loop();
    void flush_batch(std::deque<WriteItem>& batch);
    static QueryResult to_result(const Entity& e);

    Config cfg_;
    EntityPool pool_;
    GridIndex grid_;
    std::unordered_map<uint64_t, uint32_t> id_to_slot_;
    std::unordered_map<uint16_t, std::vector<uint32_t>> by_label_;

    // Flat feature storage (row-major n x feat_dim_), cache-friendly scan.
    std::vector<float> feat_flat_;
    std::vector<uint64_t> feat_ids_;
    std::unordered_map<uint64_t, uint32_t> feat_row_;
    size_t feat_dim_ = 0;

    mutable std::mutex store_mtx_;  // serializes SQLite access (flush thread + main thread)
    uint64_t next_id_ = 1;
    mutable Stats stats_{};

    std::unique_ptr<SqliteStore> store_;
    std::thread flush_thread_;
    std::mutex q_mtx_;
    std::condition_variable q_cv_;
    std::deque<WriteItem> queue_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> persisted_{0};
};

}  // namespace sm
