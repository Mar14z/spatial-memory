#include <gtest/gtest.h>

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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
