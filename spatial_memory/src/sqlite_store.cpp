#include "sm/sqlite_store.hpp"

#include <sqlite3.h>

#include <cstdio>

#include "sm/types.hpp"

namespace sm {

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
            " ts INTEGER NOT NULL,"
            " px REAL, py REAL, pz REAL,"
            " confidence REAL);")) {
        return false;
    }
    exec("CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(ts);");
    exec("CREATE INDEX IF NOT EXISTS idx_obs_eid ON observations(entity_id);");

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
            "INSERT INTO observations(entity_id,ts,px,py,pz,confidence)"
            " VALUES(?,?,?,?,?,?);",
            -1, &st_obs_, nullptr) != SQLITE_OK) {
        fprintf(stderr, "[sqlite] prepare obs failed: %s\n", sqlite3_errmsg(db_));
        return false;
    }
    return true;
}

void SqliteStore::close() {
    if (st_upsert_) { sqlite3_finalize(st_upsert_); st_upsert_ = nullptr; }
    if (st_obs_) { sqlite3_finalize(st_obs_); st_obs_ = nullptr; }
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

bool SqliteStore::append_observation(uint64_t entity_id, const Observation& o) {
    if (!st_obs_) return false;
    sqlite3_reset(st_obs_);
    sqlite3_bind_int64(st_obs_, 1, (sqlite3_int64)entity_id);
    sqlite3_bind_int64(st_obs_, 2, (sqlite3_int64)o.ts_ns);
    sqlite3_bind_double(st_obs_, 3, o.pos[0]);
    sqlite3_bind_double(st_obs_, 4, o.pos[1]);
    sqlite3_bind_double(st_obs_, 5, o.pos[2]);
    sqlite3_bind_double(st_obs_, 6, o.confidence);
    return sqlite3_step(st_obs_) == SQLITE_DONE;
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

uint64_t SqliteStore::count_observations() {
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
