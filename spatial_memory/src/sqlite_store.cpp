#include "sm/sqlite_store.hpp"

#include <sqlite3.h>

#include <cmath>
#include <cstdio>
#include <string>

#include "sm/types.hpp"

namespace sm {

namespace {

bool column_exists(sqlite3* db, const char* table, const char* col) {
    sqlite3_stmt* st = nullptr;
    bool found = false;
    std::string sql = std::string("PRAGMA table_info(") + table + ");";
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) == SQLITE_OK) {
        while (sqlite3_step(st) == SQLITE_ROW) {
            const unsigned char* name = sqlite3_column_text(st, 1);
            if (name && col == std::string(reinterpret_cast<const char*>(name))) {
                found = true;
                break;
            }
        }
    }
    sqlite3_finalize(st);
    return found;
}

}  // namespace

SqliteStore::~SqliteStore() { close(); }

bool SqliteStore::exec(const char* sql) {
    char* err = nullptr;
    int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "[sqlite] exec failed: %s\n  sql: %s\n", err ? err : "?", sql);
        sqlite3_free(err);
        return false;
    }
    return true;
}

bool SqliteStore::open(const std::string& path) {
    close();
    if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK) {
        fprintf(stderr, "[sqlite] open failed: %s\n", sqlite3_errmsg(db_));
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }
    exec("PRAGMA journal_mode=WAL;");
    exec("PRAGMA synchronous=NORMAL;");
    exec("PRAGMA busy_timeout=5000;");
    exec("PRAGMA cache_size=-16384;");

    if (!exec(
            "CREATE TABLE IF NOT EXISTS entities("
            " id INTEGER PRIMARY KEY,"
            " label INTEGER NOT NULL,"
            " state INTEGER NOT NULL,"
            " px REAL, py REAL, pz REAL,"
            " sx REAL, sy REAL, sz REAL,"
            " first_seen INTEGER NOT NULL,"
            " last_seen INTEGER NOT NULL,"
            " confidence REAL,"
            " obs_count INTEGER);")) {
        return false;
    }
    if (!exec(
            "CREATE TABLE IF NOT EXISTS observations("
            " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            " entity_id INTEGER NOT NULL,"
            " start_ts INTEGER NOT NULL,"
            " end_ts INTEGER NOT NULL,"
            " px REAL, py REAL, pz REAL,"
            " sample_count INTEGER NOT NULL DEFAULT 1,"
            " confidence REAL);")) {
        return false;
    }

    // Migrate v1 schema (ts, one row per observation) -> v2 segments.
    if (column_exists(db_, "observations", "ts") &&
        !column_exists(db_, "observations", "end_ts")) {
        fprintf(stderr, "[sqlite] migrating observations to segment schema\n");
        exec("ALTER TABLE observations RENAME COLUMN ts TO start_ts;");
        exec("ALTER TABLE observations ADD COLUMN end_ts INTEGER NOT NULL DEFAULT 0;");
        exec("UPDATE observations SET end_ts = start_ts;");
        exec("ALTER TABLE observations ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 1;");
        exec("DROP INDEX IF EXISTS idx_obs_ts;");
        exec("DROP INDEX IF EXISTS idx_obs_eid;");
    }

    exec("CREATE INDEX IF NOT EXISTS idx_seg_eid ON observations(entity_id, end_ts);");
    exec("CREATE INDEX IF NOT EXISTS idx_seg_ts ON observations(start_ts);");

    // v3 migration: feature blob column on entities
    if (!column_exists(db_, "entities", "feat")) {
        exec("ALTER TABLE entities ADD COLUMN feat BLOB;");
    }

    if (sqlite3_prepare_v2(db_,
            "INSERT INTO entities(id,label,state,px,py,pz,sx,sy,sz,"
            "first_seen,last_seen,confidence,obs_count)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET"
            " label=excluded.label,state=excluded.state,"
            " px=excluded.px,py=excluded.py,pz=excluded.pz,"
            " sx=excluded.sx,sy=excluded.sy,sz=excluded.sz,"
            " last_seen=excluded.last_seen,"
            " confidence=excluded.confidence,obs_count=excluded.obs_count;",
            -1, &st_upsert_, nullptr) != SQLITE_OK) {
        fprintf(stderr, "[sqlite] prepare upsert failed: %s\n", sqlite3_errmsg(db_));
        return false;
    }
    if (sqlite3_prepare_v2(db_,
            "SELECT seq,end_ts,px,py,pz,confidence FROM observations"
            " WHERE entity_id=? ORDER BY seq DESC LIMIT 1;",
            -1, &st_sel_last_, nullptr) != SQLITE_OK) {
        fprintf(stderr, "[sqlite] prepare sel_last failed: %s\n", sqlite3_errmsg(db_));
        return false;
    }
    if (sqlite3_prepare_v2(db_,
            "UPDATE observations SET end_ts=?, confidence=?,"
            " sample_count=sample_count+1 WHERE seq=?;",
            -1, &st_upd_seg_, nullptr) != SQLITE_OK) {
        fprintf(stderr, "[sqlite] prepare upd_seg failed: %s\n", sqlite3_errmsg(db_));
        return false;
    }
    if (sqlite3_prepare_v2(db_,
            "INSERT INTO observations(entity_id,start_ts,end_ts,px,py,pz,sample_count,confidence)"
            " VALUES(?,?,?,?,?,?,1,?);",
            -1, &st_ins_seg_, nullptr) != SQLITE_OK) {
        fprintf(stderr, "[sqlite] prepare ins_seg failed: %s\n", sqlite3_errmsg(db_));
        return false;
    }
    return true;
}

void SqliteStore::close() {
    if (st_upsert_) { sqlite3_finalize(st_upsert_); st_upsert_ = nullptr; }
    if (st_sel_last_) { sqlite3_finalize(st_sel_last_); st_sel_last_ = nullptr; }
    if (st_upd_seg_) { sqlite3_finalize(st_upd_seg_); st_upd_seg_ = nullptr; }
    if (st_ins_seg_) { sqlite3_finalize(st_ins_seg_); st_ins_seg_ = nullptr; }
    if (db_) { sqlite3_close(db_); db_ = nullptr; }
}

bool SqliteStore::begin() {
    return db_ && exec("BEGIN IMMEDIATE;");
}

bool SqliteStore::commit() {
    return db_ && exec("COMMIT;");
}

bool SqliteStore::upsert_entity(const Entity& e) {
    if (!st_upsert_) return false;
    sqlite3_reset(st_upsert_);
    sqlite3_bind_int64(st_upsert_, 1, (sqlite3_int64)e.id);
    sqlite3_bind_int(st_upsert_, 2, (int)e.label);
    sqlite3_bind_int(st_upsert_, 3, (int)e.state);
    sqlite3_bind_double(st_upsert_, 4, e.pos[0]);
    sqlite3_bind_double(st_upsert_, 5, e.pos[1]);
    sqlite3_bind_double(st_upsert_, 6, e.pos[2]);
    sqlite3_bind_double(st_upsert_, 7, e.size[0]);
    sqlite3_bind_double(st_upsert_, 8, e.size[1]);
    sqlite3_bind_double(st_upsert_, 9, e.size[2]);
    sqlite3_bind_int64(st_upsert_, 10, (sqlite3_int64)e.first_seen_ns);
    sqlite3_bind_int64(st_upsert_, 11, (sqlite3_int64)e.last_seen_ns);
    sqlite3_bind_double(st_upsert_, 12, e.confidence);
    sqlite3_bind_int64(st_upsert_, 13, (sqlite3_int64)e.obs_count);
    return sqlite3_step(st_upsert_) == SQLITE_DONE;
}

bool SqliteStore::append_segment(uint64_t entity_id, const Observation& o,
                                 int64_t max_gap_ns, float max_dist_m) {
    if (!st_sel_last_) return false;

    sqlite3_reset(st_sel_last_);
    sqlite3_bind_int64(st_sel_last_, 1, (sqlite3_int64)entity_id);

    bool merged = false;
    if (sqlite3_step(st_sel_last_) == SQLITE_ROW) {
        int64_t seq = sqlite3_column_int64(st_sel_last_, 0);
        int64_t end_ts = sqlite3_column_int64(st_sel_last_, 1);
        double lx = sqlite3_column_double(st_sel_last_, 2);
        double ly = sqlite3_column_double(st_sel_last_, 3);
        double lz = sqlite3_column_double(st_sel_last_, 4);
        double lc = sqlite3_column_double(st_sel_last_, 5);

        int64_t gap = o.ts_ns - end_ts;
        double dx = o.pos[0] - lx;
        double dy = o.pos[1] - ly;
        double dz = o.pos[2] - lz;
        double d2 = dx * dx + dy * dy + dz * dz;
        if (gap >= 0 && gap <= max_gap_ns &&
            d2 <= static_cast<double>(max_dist_m) * max_dist_m) {
            double nc = lc * 0.9 + static_cast<double>(o.confidence) * 0.1;
            sqlite3_reset(st_upd_seg_);
            sqlite3_bind_int64(st_upd_seg_, 1, (sqlite3_int64)o.ts_ns);
            sqlite3_bind_double(st_upd_seg_, 2, nc);
            sqlite3_bind_int64(st_upd_seg_, 3, seq);
            merged = (sqlite3_step(st_upd_seg_) == SQLITE_DONE);
        }
    }

    if (!merged) {
        sqlite3_reset(st_ins_seg_);
        sqlite3_bind_int64(st_ins_seg_, 1, (sqlite3_int64)entity_id);
        sqlite3_bind_int64(st_ins_seg_, 2, (sqlite3_int64)o.ts_ns);
        sqlite3_bind_int64(st_ins_seg_, 3, (sqlite3_int64)o.ts_ns);
        sqlite3_bind_double(st_ins_seg_, 4, o.pos[0]);
        sqlite3_bind_double(st_ins_seg_, 5, o.pos[1]);
        sqlite3_bind_double(st_ins_seg_, 6, o.pos[2]);
        sqlite3_bind_double(st_ins_seg_, 7, o.confidence);
        if (sqlite3_step(st_ins_seg_) != SQLITE_DONE) return false;
    }
    return true;
}

void SqliteStore::load_entities(const std::function<void(const Entity&)>& cb) {
    if (!db_) return;
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(db_,
            "SELECT id,label,state,px,py,pz,sx,sy,sz,"
            "first_seen,last_seen,confidence,obs_count"
            " FROM entities ORDER BY id;",
            -1, &st, nullptr) != SQLITE_OK) {
        return;
    }
    while (sqlite3_step(st) == SQLITE_ROW) {
        Entity e;
        e.id = (uint64_t)sqlite3_column_int64(st, 0);
        e.label = (uint16_t)sqlite3_column_int(st, 1);
        e.state = (EntityState)sqlite3_column_int(st, 2);
        e.pos[0] = (float)sqlite3_column_double(st, 3);
        e.pos[1] = (float)sqlite3_column_double(st, 4);
        e.pos[2] = (float)sqlite3_column_double(st, 5);
        e.size[0] = (float)sqlite3_column_double(st, 6);
        e.size[1] = (float)sqlite3_column_double(st, 7);
        e.size[2] = (float)sqlite3_column_double(st, 8);
        e.first_seen_ns = sqlite3_column_int64(st, 9);
        e.last_seen_ns = sqlite3_column_int64(st, 10);
        e.confidence = (float)sqlite3_column_double(st, 11);
        e.obs_count = (uint32_t)sqlite3_column_int64(st, 12);
        cb(e);
    }
    sqlite3_finalize(st);
}

namespace {
void read_history_point(sqlite3_stmt* st, HistoryPoint& hp) {
    hp.entity_id = (uint64_t)sqlite3_column_int64(st, 0);
    hp.start_ts = sqlite3_column_int64(st, 1);
    hp.end_ts = sqlite3_column_int64(st, 2);
    hp.pos[0] = (float)sqlite3_column_double(st, 3);
    hp.pos[1] = (float)sqlite3_column_double(st, 4);
    hp.pos[2] = (float)sqlite3_column_double(st, 5);
    hp.samples = (uint32_t)sqlite3_column_int64(st, 6);
    hp.confidence = (float)sqlite3_column_double(st, 7);
}
}  // namespace

void SqliteStore::query_history(uint64_t entity_id, int64_t from_ts, int64_t to_ts,
                                size_t limit,
                                const std::function<void(const HistoryPoint&)>& cb) const {
    if (!db_) return;
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(db_,
            "SELECT entity_id,start_ts,end_ts,px,py,pz,sample_count,confidence"
            " FROM observations"
            " WHERE entity_id=? AND end_ts>=? AND start_ts<=?"
            " ORDER BY start_ts ASC LIMIT ?;",
            -1, &st, nullptr) != SQLITE_OK) {
        return;
    }
    sqlite3_bind_int64(st, 1, (sqlite3_int64)entity_id);
    sqlite3_bind_int64(st, 2, (sqlite3_int64)from_ts);
    sqlite3_bind_int64(st, 3, (sqlite3_int64)to_ts);
    sqlite3_bind_int64(st, 4, (sqlite3_int64)limit);
    while (sqlite3_step(st) == SQLITE_ROW) {
        HistoryPoint hp;
        read_history_point(st, hp);
        cb(hp);
    }
    sqlite3_finalize(st);
}

void SqliteStore::query_time_range(int64_t from_ts, int64_t to_ts, int label, size_t limit,
                                   const std::function<void(const HistoryPoint&)>& cb) const {
    if (!db_) return;
    const char* sql_label =
        "SELECT o.entity_id,o.start_ts,o.end_ts,o.px,o.py,o.pz,o.sample_count,o.confidence"
        " FROM observations o JOIN entities e ON e.id=o.entity_id"
        " WHERE o.end_ts>=? AND o.start_ts<=? AND e.label=?"
        " ORDER BY o.start_ts ASC LIMIT ?;";
    const char* sql_all =
        "SELECT o.entity_id,o.start_ts,o.end_ts,o.px,o.py,o.pz,o.sample_count,o.confidence"
        " FROM observations o"
        " WHERE o.end_ts>=? AND o.start_ts<=?"
        " ORDER BY o.start_ts ASC LIMIT ?;";

    sqlite3_stmt* st = nullptr;
    if (label >= 0) {
        if (sqlite3_prepare_v2(db_, sql_label, -1, &st, nullptr) != SQLITE_OK) return;
        sqlite3_bind_int64(st, 1, (sqlite3_int64)from_ts);
        sqlite3_bind_int64(st, 2, (sqlite3_int64)to_ts);
        sqlite3_bind_int(st, 3, label);
        sqlite3_bind_int64(st, 4, (sqlite3_int64)limit);
    } else {
        if (sqlite3_prepare_v2(db_, sql_all, -1, &st, nullptr) != SQLITE_OK) return;
        sqlite3_bind_int64(st, 1, (sqlite3_int64)from_ts);
        sqlite3_bind_int64(st, 2, (sqlite3_int64)to_ts);
        sqlite3_bind_int64(st, 3, (sqlite3_int64)limit);
    }
    while (sqlite3_step(st) == SQLITE_ROW) {
        HistoryPoint hp;
        read_history_point(st, hp);
        cb(hp);
    }
    sqlite3_finalize(st);
}

bool SqliteStore::set_feature(uint64_t entity_id, const float* feat, size_t dim) {
    if (!db_ || !feat || dim == 0) return false;
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(db_, "UPDATE entities SET feat=? WHERE id=?;", -1, &st, nullptr) != SQLITE_OK) {
        return false;
    }
    sqlite3_bind_blob(st, 1, feat, (int)(dim * sizeof(float)), SQLITE_TRANSIENT);
    sqlite3_bind_int64(st, 2, (sqlite3_int64)entity_id);
    bool ok = sqlite3_step(st) == SQLITE_DONE;
    sqlite3_finalize(st);
    return ok;
}

void SqliteStore::load_features(
    const std::function<void(uint64_t, const float*, size_t)>& cb) const {
    if (!db_) return;
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT id, feat FROM entities WHERE feat IS NOT NULL;",
                           -1, &st, nullptr) != SQLITE_OK) {
        return;
    }
    while (sqlite3_step(st) == SQLITE_ROW) {
        uint64_t id = (uint64_t)sqlite3_column_int64(st, 0);
        const void* blob = sqlite3_column_blob(st, 1);
        int bytes = sqlite3_column_bytes(st, 1);
        if (blob && bytes > 0) {
            cb(id, static_cast<const float*>(blob), (size_t)bytes / sizeof(float));
        }
    }
    sqlite3_finalize(st);
}

uint64_t SqliteStore::count_features() const {
    if (!db_) return 0;
    sqlite3_stmt* st = nullptr;
    uint64_t n = 0;
    if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM entities WHERE feat IS NOT NULL;",
                           -1, &st, nullptr) == SQLITE_OK) {
        if (sqlite3_step(st) == SQLITE_ROW) n = (uint64_t)sqlite3_column_int64(st, 0);
    }
    sqlite3_finalize(st);
    return n;
}

uint64_t SqliteStore::count_observations() const {
    if (!db_) return 0;
    sqlite3_stmt* st = nullptr;
    uint64_t n = 0;
    if (sqlite3_prepare_v2(db_, "SELECT COALESCE(SUM(sample_count),0) FROM observations;",
                           -1, &st, nullptr) == SQLITE_OK) {
        if (sqlite3_step(st) == SQLITE_ROW) n = (uint64_t)sqlite3_column_int64(st, 0);
    }
    sqlite3_finalize(st);
    return n;
}

uint64_t SqliteStore::count_segments() const {
    if (!db_) return 0;
    sqlite3_stmt* st = nullptr;
    uint64_t n = 0;
    if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM observations;", -1, &st, nullptr) == SQLITE_OK) {
        if (sqlite3_step(st) == SQLITE_ROW) n = (uint64_t)sqlite3_column_int64(st, 0);
    }
    sqlite3_finalize(st);
    return n;
}

}  // namespace sm
