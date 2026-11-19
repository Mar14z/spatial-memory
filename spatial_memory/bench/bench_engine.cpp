#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "sm/engine.hpp"

using namespace sm;
using Clock = std::chrono::steady_clock;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

static double rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string key;
    while (f >> key) {
        if (key == "VmRSS:") {
            double kb = 0;
            f >> kb;
            return kb / 1024.0;
        }
    }
    return -1;
}

static double percentile(std::vector<double>& v, double p) {
    size_t i = (size_t)std::min(v.size() - 1, (size_t)(v.size() * p));
    std::nth_element(v.begin(), v.begin() + i, v.end());
    return v[i];
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-20.0f, 20.0f);

    // ---------- 1) pure insert ----------
    {
        SpatialMemoryEngine e;
        const int N = 100000;
        double rss0 = rss_mb();
        auto t0 = Clock::now();
        for (int i = 0; i < N; i++) {
            Observation o;
            o.label = (uint16_t)(i % 256);
            o.pos[0] = U(rng);
            o.pos[1] = U(rng);
            o.pos[2] = U(rng);
            o.ts_ns = (int64_t)i;
            o.confidence = 0.9f;
            e.add_observation(o);
        }
        auto t1 = Clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        printf("[insert]  entities=%zu  %.0f k/s  rss_delta=%.1f MB  cells=%zu\n",
               e.size(), N / secs / 1000.0, rss_mb() - rss0, e.grid_cells());

        // ---------- 2) radius query latency ----------
        const int Q = 10000;
        std::vector<double> lat(Q);
        long long total = 0;
        for (int q = 0; q < Q; q++) {
            float c[3] = {U(rng), U(rng), U(rng)};
            auto s0 = Clock::now();
            auto r = e.query_radius(c, 1.5f);
            auto s1 = Clock::now();
            lat[q] = std::chrono::duration<double, std::micro>(s1 - s0).count();
            total += (long long)r.size();
        }
        printf("[query]   r=1.5m  P50=%.1f us  P99=%.1f us  max=%.1f us  avg_hits=%.1f\n",
               percentile(lat, 0.50), percentile(lat, 0.99),
               *std::max_element(lat.begin(), lat.end()), (double)total / Q);
    }

    // ---------- 3) association: 100 objects x 10 frames, 5cm noise ----------
    {
        SpatialMemoryEngine e;
        std::normal_distribution<float> noise(0.0f, 0.05f);
        const int OBJ = 100, FRAMES = 10;
        float base[OBJ][3];
        for (int i = 0; i < OBJ; i++) {
            base[i][0] = (float)(i % 10) * 2.0f;
            base[i][1] = (float)((i / 10) % 10) * 2.0f;
            base[i][2] = (float)(i / 100) * 2.0f;
        }
        for (int f = 0; f < FRAMES; f++) {
            for (int i = 0; i < OBJ; i++) {
                Observation o;
                o.label = 1;
                o.pos[0] = base[i][0] + noise(rng);
                o.pos[1] = base[i][1] + noise(rng);
                o.pos[2] = base[i][2] + noise(rng);
                o.ts_ns = (int64_t)f * 100'000'000LL;
                o.confidence = 0.8f;
                e.add_observation(o);
            }
        }
        auto s = e.stats();
        printf("[assoc]   %d obj x %d frames -> entities=%zu (expect %d), updates=%llu\n",
               OBJ, FRAMES, e.size(), OBJ, (unsigned long long)s.updates);
    }

    // ---------- 4) moved object leaves no ghost (stepwise, crosses many cells) ----------
    {
        SpatialMemoryEngine e;
        Observation o;
        o.label = 1;
        o.pos[0] = 0.5f; o.pos[1] = 0.5f; o.pos[2] = 0.5f;
        o.ts_ns = 0;
        e.add_observation(o);
        for (int i = 0; i < 20; i++) {
            o.pos[0] += 0.4f;
            o.ts_ns += 100'000'000LL;
            e.add_observation(o);
        }
        float oldp[3] = {0.5f, 0.5f, 0.5f};
        float newp[3] = {o.pos[0], 0.5f, 0.5f};
        printf("[move]    entities=%zu (expect 1)  old_hit=%zu (expect 0)  new_hit=%zu (expect 1)\n",
               e.size(), e.query_radius(oldp, 0.6f).size(), e.query_radius(newp, 0.6f).size());
    }

    return 0;
}
