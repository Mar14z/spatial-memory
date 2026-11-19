#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "sm/types.hpp"

namespace sm {

class EntityPool {
public:
    uint32_t alloc() {
        if (!free_list_.empty()) {
            uint32_t idx = free_list_.back();
            free_list_.pop_back();
            data_[idx] = Entity{};
            return idx;
        }
        data_.emplace_back();
        return static_cast<uint32_t>(data_.size() - 1);
    }

    void free_slot(uint32_t idx) {
        data_[idx].state = EntityState::kDeleted;
        free_list_.push_back(idx);
    }

    Entity& get(uint32_t idx) { return data_[idx]; }
    const Entity& get(uint32_t idx) const { return data_[idx]; }

    size_t capacity() const { return data_.size(); }
    size_t live_count() const { return data_.size() - free_list_.size(); }

private:
    std::vector<Entity> data_;
    std::vector<uint32_t> free_list_;
};

}  // namespace sm
