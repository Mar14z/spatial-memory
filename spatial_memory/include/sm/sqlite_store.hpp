#pragma once
#include <cstdint>
#include <functional>
#include <string>

struct sqlite3;
struct sqlite3_stmt;

namespace sm {

struct Entity;
struct Observation;

// One compressed "resident segment" on the observation timeline:
// a run of observations of the same object, spatially stable (<5cm)
// and temporally contiguous (<5s).
struct HistoryPoint {
    uint64_t entity_id = 0;
    int64_t start_ts = 0;
    int64_t end_ts = 0;
    float pos[3] = {0.f, 0.f, 0.f};
    uint32_t samples = 0;
    float confidence = 0.f;
};

// Thin wrapper over SQLite (WAL mode) for the cold/persistent layer.
class SqliteStore {
public:
    SqliteStore() = default;
    ~SqliteStore();
    SqliteStore(const SqliteStore&) = delete;
    SqliteStore& operator=(const SqliteStore&) = delete;

    bool open(const std::string& path);
    void close();
    bool is_open() const { return db_ != nullptr; }

    bool begin();
    bool commit();

    bool upsert_entity(const Entity& e);
    // v2: merge into the last segment when within gap/dist, else append a new segment.
    bool append_segment(uint64_t entity_id, const Observation& o,
                        int64_t max_gap_ns, float max_dist_m);

    void load_entities(const std::function<void(const Entity&)>& cb);

    void query_history(uint64_t entity_id, int64_t from_ts, int64_t to_ts, size_t limit,
                       const std::function<void(const HistoryPoint&)>& cb) const;
    // label < 0 means no label filter
    void query_time_range(int64_t from_ts, int64_t to_ts, int label, size_t limit,
                          const std::function<void(const HistoryPoint&)>& cb) const;

    uint64_t count_observations() const;  // SUM(sample_count)
    uint64_t count_segments() const;      // COUNT(*)

    // v3: per-object semantic feature (e.g. CLIP 512-d fp32 blob)
    bool set_feature(uint64_t entity_id, const float* feat, size_t dim);
    void load_features(const std::function<void(uint64_t, const float*, size_t)>& cb) const;
    uint64_t count_features() const;

    // v3d: entity lifecycle. Writes a tombstone (state=2, feat=NULL) so the timeline
    // stays resolvable and the id watermark survives restarts.
    bool delete_entity(uint64_t entity_id);
    uint64_t max_entity_id() const;

private:
    bool exec(const char* sql);

    sqlite3* db_ = nullptr;
    sqlite3_stmt* st_upsert_ = nullptr;
    sqlite3_stmt* st_sel_last_ = nullptr;
    sqlite3_stmt* st_upd_seg_ = nullptr;
    sqlite3_stmt* st_ins_seg_ = nullptr;
};

}  // namespace sm
