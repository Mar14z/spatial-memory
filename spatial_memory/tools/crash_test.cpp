// Crash-recovery test tool.
//   crash_test write  <db>   : ingest observations forever (~2k/s), print progress to stderr
//   crash_test verify <db>   : reopen the db, recover indexes, run SQLite integrity check
//
// Usage (see crash_test.sh):
//   ./crash_test write /tmp/sm_crash.db &   # then kill -9 it
//   ./crash_test verify /tmp/sm_crash.db
#include <sqlite3.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "sm/engine.hpp"

using namespace sm;

static Observation make_obs(int i) {
    Observation o;
    o.label = (uint16_t)(i % 16);
    o.pos[0] = (float)(i % 100) * 0.5f;
    o.pos[1] = (float)((i / 100) % 100) * 0.5f;
    o.pos[2] = (float)((i / 10000) % 10) * 0.5f;
    o.ts_ns = (int64_t)i * 10'000'000LL;
    o.confidence = 0.9f;
    return o;
}

static int run_verify(const std::string& db) {
    Config c;
    c.db_path = db;
    SpatialMemoryEngine e(c);
    size_t ents = e.size();
    uint64_t obs = e.persisted_observations();
    uint64_t segs = e.persisted_segments();
    float ctr[3] = {0.f, 0.f, 0.f};
    auto hits = e.query_radius(ctr, 50.f, std::nullopt);
    auto hist = e.query_history(1, 0, INT64_MAX, 1000);
    printf("restored_entities=%zu obs=%llu segments=%llu radius_hits=%zu history_1=%zu\n",
           ents, (unsigned long long)obs, (unsigned long long)segs,
           hits.size(), hist.size());

    int integrity_ok = -1;
    sqlite3* dbh = nullptr;
    if (sqlite3_open(db.c_str(), &dbh) == SQLITE_OK) {
        sqlite3_stmt* st = nullptr;
        if (sqlite3_prepare_v2(dbh, "PRAGMA integrity_check;", -1, &st, nullptr) == SQLITE_OK) {
            if (sqlite3_step(st) == SQLITE_ROW) {
                const unsigned char* s = sqlite3_column_text(st, 0);
                integrity_ok = (s && strcmp(reinterpret_cast<const char*>(s), "ok") == 0) ? 1 : 0;
                if (!integrity_ok) printf("integrity_detail=%s\n", s ? reinterpret_cast<const char*>(s) : "?");
            }
        }
        sqlite3_finalize(st);
        sqlite3_close(dbh);
    }
    printf("integrity_check=%s\n", integrity_ok == 1 ? "ok" : "FAIL");

    int fail = 0;
    if (ents == 0 || obs == 0) {
        printf("no data recovered\n");
        fail = 1;
    }
    if (integrity_ok != 1) fail = 1;
    printf(fail ? "VERIFY_FAIL\n" : "VERIFY_OK\n");
    return fail;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s write|verify <db_path>\n", argv[0]);
        return 2;
    }
    std::string mode = argv[1];
    std::string db = argv[2];

    if (mode == "write") {
        Config c;
        c.db_path = db;
        SpatialMemoryEngine e(c);
        int i = 0;
        while (true) {
            e.add_observation(make_obs(i++));
            if (i % 100 == 0) {
                fprintf(stderr, "written=%d\n", i);
                fflush(stderr);
                usleep(50000);  // ~2k obs/s
            }
        }
    }
    if (mode == "verify") return run_verify(db);

    fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
}
