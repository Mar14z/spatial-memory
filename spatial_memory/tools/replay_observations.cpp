// Replay Observation JSONL streams through the engine.
// Used to drive the engine with dataset-derived observations (ReplicaCAD etc.)
// without any camera/NPU dependency.
//
// Usage:
//   replay_observations <a.jsonl> [b.jsonl ...] [--db PATH] [--gap SEC]
//                       [--stale SEC] [--forget SEC] [--query N]
//
// Consecutive files are treated as separate sessions separated by --gap seconds;
// entities older than --forget are swept right before the next session starts.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "sm/engine.hpp"

using namespace sm;

namespace {

const char* field(const std::string& line, const char* key) {
    const char* p = strstr(line.c_str(), key);
    return p ? p + strlen(key) : nullptr;
}

bool parse_line(const std::string& line, Observation& o) {
    const char* p = nullptr;
    if (!(p = field(line, "\"label\":"))) return false;
    o.label = (uint16_t)atoi(p);
    if (!(p = field(line, "\"x\":"))) return false;
    o.pos[0] = strtof(p, nullptr);
    if (!(p = field(line, "\"y\":"))) return false;
    o.pos[1] = strtof(p, nullptr);
    if (!(p = field(line, "\"z\":"))) return false;
    o.pos[2] = strtof(p, nullptr);
    if (!(p = field(line, "\"sx\":"))) return false;
    o.size[0] = strtof(p, nullptr);
    if (!(p = field(line, "\"sy\":"))) return false;
    o.size[1] = strtof(p, nullptr);
    if (!(p = field(line, "\"sz\":"))) return false;
    o.size[2] = strtof(p, nullptr);
    if (!(p = field(line, "\"ts\":"))) return false;
    o.ts_ns = strtoll(p, nullptr, 10);
    if (!(p = field(line, "\"conf\":"))) return false;
    o.confidence = strtof(p, nullptr);
    return true;
}

double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    size_t i = (size_t)std::min(v.size() - 1, (size_t)(v.size() * p));
    std::nth_element(v.begin(), v.begin() + (ptrdiff_t)i, v.end());
    return v[i];
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> inputs;
    Config cfg;
    double gap_s = 0.0;
    size_t query_n = 200;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--db") cfg.db_path = next();
        else if (a == "--gap") gap_s = atof(next().c_str());
        else if (a == "--stale") cfg.stale_after_ns = (int64_t)(atof(next().c_str()) * 1e9);
        else if (a == "--forget") cfg.forget_after_ns = (int64_t)(atof(next().c_str()) * 1e9);
        else if (a == "--query") query_n = (size_t)atoi(next().c_str());
        else inputs.push_back(a);
    }
    if (inputs.empty()) {
        fprintf(stderr,
                "usage: replay_observations <a.jsonl> [b.jsonl ...] [--db PATH] [--gap SEC] "
                "[--stale SEC] [--forget SEC] [--query N]\n");
        return 2;
    }

    SpatialMemoryEngine e(cfg);
    const int64_t gap_ns = (int64_t)(gap_s * 1e9);
    int64_t offset = 0;
    int64_t last_ts = 0;
    std::vector<uint64_t> ids;
    uint64_t lines_total = 0;
    int bad_lines = 0;

    for (size_t fi = 0; fi < inputs.size(); fi++) {
        if (fi > 0) {
            offset += gap_ns;
            if (gap_ns > 0 && cfg.forget_after_ns > 0) e.sweep(last_ts + gap_ns - 1);
        }
        std::ifstream in(inputs[fi]);
        if (!in) {
            fprintf(stderr, "cannot open %s\n", inputs[fi].c_str());
            return 1;
        }
        auto t0 = std::chrono::steady_clock::now();
        std::string line;
        uint64_t n = 0;
        int64_t frames = 0;
        int64_t frame_ts = -1;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            Observation o;
            if (!parse_line(line, o)) {
                bad_lines++;
                continue;
            }
            o.ts_ns += offset;
            if (o.ts_ns != frame_ts) {
                frame_ts = o.ts_ns;
                frames++;
            }
            ids.push_back(e.add_observation(o));
            n++;
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_ts = std::max(last_ts, frame_ts);
        lines_total += n;
        auto s = e.stats();
        printf(
            "[%s] obs=%llu frames=%lld entities=%zu new=%llu upd=%llu forgotten=%llu "
            "%.1fms (%.0f obs/s)\n",
            inputs[fi].c_str(), (unsigned long long)n, (long long)frames, e.size(),
            (unsigned long long)s.new_entities, (unsigned long long)s.updates,
            (unsigned long long)s.forgotten, ms, ms > 0 ? n / (ms / 1000.0) : 0.0);
    }

    // Query latency on random known entities.
    std::mt19937 rng(123);
    std::vector<double> us;
    us.reserve(query_n);
    for (size_t i = 0; i < query_n && !ids.empty(); i++) {
        uint64_t id = ids[rng() % ids.size()];
        auto r = e.get(id);
        if (!r) continue;
        float c[3] = {r->pos[0], r->pos[1], r->pos[2]};
        auto t0 = std::chrono::steady_clock::now();
        auto hits = e.query_radius(c, 1.5f);
        auto t1 = std::chrono::steady_clock::now();
        (void)hits;
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    auto st = e.stats();
    printf(
        "total obs=%llu lines=%llu bad=%d entities=%zu grid_cells=%zu "
        "new=%llu upd=%llu forgotten=%llu features=%zu\n",
        (unsigned long long)lines_total, (unsigned long long)lines_total, bad_lines, e.size(),
        e.grid_cells(), (unsigned long long)st.new_entities, (unsigned long long)st.updates,
        (unsigned long long)st.forgotten, e.feature_count());
    if (!us.empty()) {
        printf("query_radius r=1.5m n=%zu P50=%.1fus P99=%.1fus\n", us.size(), percentile(us, 0.5),
               percentile(us, 0.99));
    }
    return 0;
}
