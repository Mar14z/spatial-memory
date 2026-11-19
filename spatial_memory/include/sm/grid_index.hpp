#pragma once
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sm {

class GridIndex {
public:
    explicit GridIndex(float cell_size) : cell_(cell_size) {}

    void insert(uint32_t slot, const float p[3]) {
        cells_[key(p)].push_back(slot);
    }

    bool remove(uint32_t slot, const float p[3]) {
        auto it = cells_.find(key(p));
        if (it == cells_.end()) return false;
        auto& v = it->second;
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i] == slot) {
                v[i] = v.back();
                v.pop_back();
                if (v.empty()) cells_.erase(it);
                return true;
            }
        }
        return false;
    }

    template <class F>
    void for_candidates(const float c[3], float r, F&& f) const {
        int32_t x0 = cell_idx(c[0] - r), x1 = cell_idx(c[0] + r);
        int32_t y0 = cell_idx(c[1] - r), y1 = cell_idx(c[1] + r);
        int32_t z0 = cell_idx(c[2] - r), z1 = cell_idx(c[2] + r);
        // Large-radius guard: scanning r^3 empty cells is worse than
        // iterating only the occupied ones.
        int64_t nx = (int64_t)x1 - x0 + 1;
        int64_t ny = (int64_t)y1 - y0 + 1;
        int64_t nz = (int64_t)z1 - z0 + 1;
        if (nx * ny * nz > 262144) {
            for (const auto& kv : cells_) {
                for (uint32_t s : kv.second) f(s);
            }
            return;
        }
        for (int32_t x = x0; x <= x1; ++x)
            for (int32_t y = y0; y <= y1; ++y)
                for (int32_t z = z0; z <= z1; ++z) {
                    auto it = cells_.find(pack(x, y, z));
                    if (it == cells_.end()) continue;
                    for (uint32_t s : it->second) f(s);
                }
    }

    size_t cell_count() const { return cells_.size(); }

private:
    int32_t cell_idx(float v) const {
        return static_cast<int32_t>(std::floor(v / cell_));
    }

    static constexpr uint64_t kOff = 1ull << 20;

    static uint64_t pack(int32_t x, int32_t y, int32_t z) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(x + static_cast<int32_t>(kOff))) << 42)
             | (static_cast<uint64_t>(static_cast<uint32_t>(y + static_cast<int32_t>(kOff))) << 21)
             |  static_cast<uint64_t>(static_cast<uint32_t>(z + static_cast<int32_t>(kOff)));
    }

    uint64_t key(const float p[3]) const {
        return pack(cell_idx(p[0]), cell_idx(p[1]), cell_idx(p[2]));
    }

    float cell_;
    std::unordered_map<uint64_t, std::vector<uint32_t>> cells_;
};

}  // namespace sm
