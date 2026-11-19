#include <gtest/gtest.h>

#include <cstdio>
#include <string>

#include "sm/engine.hpp"

using namespace sm;

namespace {
Observation make_obs(uint16_t label, float x, float y, float z, int64_t ts_ns,
                     float conf = 0.9f) {
    Observation o;
    o.label = label;
    o.pos[0] = x;
    o.pos[1] = y;
    o.pos[2] = z;
    o.ts_ns = ts_ns;
    o.confidence = conf;
    return o;
}
}  // namespace

TEST(EngineTest, SameObjectMergedWithinWindow) {
    SpatialMemoryEngine e;
    e.add_observation(make_obs(1, 1.0f, 0.0f, 0.0f, 0));
    e.add_observation(make_obs(1, 1.05f, 0.0f, 0.0f, 100'000'000LL));
    EXPECT_EQ(e.size(), 1u);
    auto r = e.query_label(1);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].id, 1u);
}

TEST(EngineTest, DifferentLabelsNotMerged) {
    SpatialMemoryEngine e;
    e.add_observation(make_obs(1, 1.0f, 0.0f, 0.0f, 0));
    e.add_observation(make_obs(2, 1.0f, 0.0f, 0.0f, 0));
    EXPECT_EQ(e.size(), 2u);
}

TEST(EngineTest, TooFarNotMerged) {
    SpatialMemoryEngine e;
    e.add_observation(make_obs(1, 0.0f, 0.0f, 0.0f, 0));
    e.add_observation(make_obs(1, 3.0f, 0.0f, 0.0f, 0));
    EXPECT_EQ(e.size(), 2u);
}

TEST(EngineTest, ExpiredWindowCreatesNew) {
    SpatialMemoryEngine e;
    e.add_observation(make_obs(1, 1.0f, 0.0f, 0.0f, 0));
    // 6s later > 5s window
    e.add_observation(make_obs(1, 1.0f, 0.0f, 0.0f, 6'000'000'000LL));
    EXPECT_EQ(e.size(), 2u);
}

TEST(EngineTest, MovedObjectFoundAtNewPositionOnly) {
    SpatialMemoryEngine e;
    float x = 0.5f;
    e.add_observation(make_obs(1, x, 0.5f, 0.5f, 0));
    // realistic motion: 0.4m per frame (within 0.5m assoc radius), crosses grid cells
    int64_t ts = 0;
    for (int i = 0; i < 12; i++) {
        x += 0.4f;
        ts += 100'000'000LL;
        e.add_observation(make_obs(1, x, 0.5f, 0.5f, ts));
    }
    EXPECT_EQ(e.size(), 1u);

    float old_p[3] = {0.5f, 0.5f, 0.5f};
    EXPECT_TRUE(e.query_radius(old_p, 0.5f).empty());

    float new_p[3] = {x, 0.5f, 0.5f};
    auto r = e.query_radius(new_p, 0.5f);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_NEAR(r[0].pos[0], x, 1e-5f);
}

TEST(EngineTest, QueryRadiusBoundary) {
    SpatialMemoryEngine e;
    e.add_observation(make_obs(1, 1.0f, 0.0f, 0.0f, 0));
    float c[3] = {0.0f, 0.0f, 0.0f};
    EXPECT_EQ(e.query_radius(c, 0.99f).size(), 0u);
    EXPECT_EQ(e.query_radius(c, 1.01f).size(), 1u);
}

TEST(EngineTest, QueryRadiusWithLabelFilter) {
    SpatialMemoryEngine e;
    e.add_observation(make_obs(1, 0.2f, 0.0f, 0.0f, 0));
    e.add_observation(make_obs(2, 0.3f, 0.0f, 0.0f, 0));
    float c[3] = {0.0f, 0.0f, 0.0f};
    EXPECT_EQ(e.query_radius(c, 1.0f).size(), 2u);
    EXPECT_EQ(e.query_radius(c, 1.0f, 1).size(), 1u);
    EXPECT_EQ(e.query_radius(c, 1.0f, 2).size(), 1u);
    EXPECT_EQ(e.query_radius(c, 1.0f, 3).size(), 0u);
}

TEST(EngineTest, GetById) {
    SpatialMemoryEngine e;
    uint64_t id = e.add_observation(make_obs(7, 2.0f, 3.0f, 4.0f, 0));
    auto r = e.get(id);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->label, 7u);
    EXPECT_FALSE(e.get(9999).has_value());
}

TEST(EngineTest, StatsCounts) {
    SpatialMemoryEngine e;
    e.add_observation(make_obs(1, 1.0f, 0.0f, 0.0f, 0));
    e.add_observation(make_obs(1, 1.02f, 0.0f, 0.0f, 50'000'000LL));
    auto s = e.stats();
    EXPECT_EQ(s.adds, 2u);
    EXPECT_EQ(s.new_entities, 1u);
    EXPECT_EQ(s.updates, 1u);
}

TEST(PoolTest, FreeListReuse) {
    EntityPool p;
    uint32_t a = p.alloc();
    uint32_t b = p.alloc();
    p.free_slot(a);
    uint32_t c = p.alloc();
    EXPECT_EQ(c, a);
    EXPECT_EQ(p.live_count(), 2u);
    (void)b;
}

TEST(PersistTest, RoundTripRestore) {
    std::string base = "/tmp/sm_test_persist";
    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());

    {
        Config c;
        c.db_path = base;
        SpatialMemoryEngine e(c);
        ASSERT_TRUE(e.persistent());
        e.add_observation(make_obs(1, 1.0f, 2.0f, 3.0f, 0));
        e.add_observation(make_obs(1, 1.02f, 2.0f, 3.0f, 50'000'000LL));
        e.add_observation(make_obs(2, 5.0f, 0.0f, 0.0f, 0));
        EXPECT_EQ(e.size(), 2u);
    }  // destructor flushes remaining queue

    Config c2;
    c2.db_path = base;
    SpatialMemoryEngine e2(c2);
    EXPECT_EQ(e2.size(), 2u);

    float p[3] = {1.0f, 2.0f, 3.0f};
    auto r = e2.query_radius(p, 1.0f);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].label, 1u);
    EXPECT_EQ(e2.persisted_observations(), 3u);

    // id continuity after recovery (must not collide with restored ids)
    uint64_t nid = e2.add_observation(make_obs(3, 9.0f, 9.0f, 9.0f, 0));
    EXPECT_GT(nid, 2u);

    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());
}

TEST(PersistTest, MemoryModeUnaffected) {
    Config c;  // no db_path -> pure in-memory
    SpatialMemoryEngine e(c);
    EXPECT_FALSE(e.persistent());
    EXPECT_EQ(e.persisted_observations(), 0u);
    e.add_observation(make_obs(1, 0.f, 0.f, 0.f, 0));
    EXPECT_EQ(e.size(), 1u);
}

TEST(SegmentTest, StaticObjectCompresses) {
    std::string base = "/tmp/sm_test_seg1";
    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());
    {
        Config c;
        c.db_path = base;
        SpatialMemoryEngine e(c);
        for (int i = 0; i < 100; i++) {
            e.add_observation(make_obs(1, 1.0f + 0.01f * (float)(i % 3), 0.f, 0.f,
                                       (int64_t)i * 100'000'000LL));
        }
    }
    Config c2;
    c2.db_path = base;
    SpatialMemoryEngine e2(c2);
    EXPECT_EQ(e2.persisted_observations(), 100u);  // SUM(sample_count)
    EXPECT_EQ(e2.persisted_segments(), 1u);        // 100 obs -> 1 resident segment
    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());
}

TEST(SegmentTest, GapSplitsSegments) {
    std::string base = "/tmp/sm_test_seg2";
    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());
    {
        Config c;
        c.db_path = base;
        c.segment_max_gap_ns = 1'000'000'000LL;  // 1s merge window
        SpatialMemoryEngine e(c);
        e.add_observation(make_obs(1, 1.f, 0.f, 0.f, 0));
        // 2s gap: still same entity (5s assoc window) but > 1s segment gap -> new segment
        e.add_observation(make_obs(1, 1.f, 0.f, 0.f, 2'000'000'000LL));
    }
    Config c2;
    c2.db_path = base;
    SpatialMemoryEngine e2(c2);
    EXPECT_EQ(e2.size(), 1u);
    EXPECT_EQ(e2.persisted_segments(), 2u);
    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());
}

TEST(SegmentTest, MoveSplitsSegments) {
    std::string base = "/tmp/sm_test_seg3";
    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());
    {
        Config c;
        c.db_path = base;
        SpatialMemoryEngine e(c);
        for (int i = 0; i < 10; i++) {
            e.add_observation(make_obs(1, 1.f + 0.1f * (float)i, 0.f, 0.f,
                                       (int64_t)i * 100'000'000LL));
        }
    }
    Config c2;
    c2.db_path = base;
    SpatialMemoryEngine e2(c2);
    EXPECT_EQ(e2.size(), 1u);                   // still one entity
    EXPECT_EQ(e2.persisted_segments(), 10u);    // each 0.1m step > 5cm -> new segment
    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());
}

TEST(QueryTest, HistoryAndTimeRange) {
    std::string base = "/tmp/sm_test_query";
    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());
    {
        Config c;
        c.db_path = base;
        SpatialMemoryEngine e(c);
        e.add_observation(make_obs(1, 1.f, 0.f, 0.f, 0));
        e.add_observation(make_obs(1, 1.f, 0.f, 0.f, 1'000'000'000LL));
        e.add_observation(make_obs(2, 5.f, 0.f, 0.f, 1'000'000'000LL));
    }
    Config c2;
    c2.db_path = base;
    SpatialMemoryEngine e2(c2);

    auto h = e2.query_history(1, 0, 2'000'000'000LL, 100);
    ASSERT_EQ(h.size(), 1u);  // merged segment
    EXPECT_EQ(h[0].samples, 2u);
    EXPECT_EQ(h[0].start_ts, 0);
    EXPECT_EQ(h[0].end_ts, 1'000'000'000LL);

    auto t = e2.query_time_range(0, 2'000'000'000LL, std::nullopt, 100);
    EXPECT_EQ(t.size(), 2u);  // one segment per entity

    auto t1 = e2.query_time_range(0, 2'000'000'000LL, 2, 100);
    ASSERT_EQ(t1.size(), 1u);
    EXPECT_EQ(t1[0].entity_id, 2u);

    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());
}

TEST(FeatureTest, SetQueryAndPersist) {
    std::string base = "/tmp/sm_test_feat";
    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());

    uint64_t a = 0, b = 0;
    {
        Config c;
        c.db_path = base;
        SpatialMemoryEngine e(c);
        a = e.add_observation(make_obs(1, 0.f, 0.f, 0.f, 0));
        b = e.add_observation(make_obs(2, 5.f, 0.f, 0.f, 0));

        std::vector<float> fa(512, 0.f), fb(512, 0.f);
        fa[0] = 3.f;   // not normalized on purpose, engine must normalize
        fb[1] = 2.f;
        EXPECT_TRUE(e.set_feature(a, fa.data(), 512));
        EXPECT_TRUE(e.set_feature(b, fb.data(), 512));
        EXPECT_EQ(e.feature_count(), 2u);

        auto hits = e.query_by_feature(fa.data(), 512, 2);
        ASSERT_EQ(hits.size(), 2u);
        EXPECT_EQ(hits[0].id, a);
        EXPECT_NEAR(hits[0].score, 1.0f, 1e-4f);
        EXPECT_NEAR(hits[1].score, 0.0f, 1e-4f);
    }

    // reopen: features must be restored from SQLite blob
    Config c2;
    c2.db_path = base;
    SpatialMemoryEngine e2(c2);
    EXPECT_EQ(e2.feature_count(), 2u);
    std::vector<float> q(512, 0.f);
    q[1] = 1.f;
    auto h = e2.query_by_feature(q.data(), 512, 1);
    ASSERT_EQ(h.size(), 1u);
    EXPECT_EQ(h[0].id, b);
    EXPECT_EQ(h[0].label, 2u);

    for (auto suf : {"", "-wal", "-shm"}) std::remove((base + std::string(suf)).c_str());
}

TEST(FeatureTest, RejectsUnknownEntity) {
    SpatialMemoryEngine e;
    std::vector<float> f(512, 0.f);
    f[0] = 1.f;
    EXPECT_FALSE(e.set_feature(999, f.data(), 512));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
