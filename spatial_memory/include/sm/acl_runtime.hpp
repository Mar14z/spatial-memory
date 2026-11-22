#pragma once
#include <mutex>

namespace sm {

// Process-wide Ascend ACL runtime shared by the CLIP vision/text towers.
// aclInit/aclFinalize are reference counted, so destroying one extractor can no
// longer tear down the device context still in use by the other.
class AclRuntime {
public:
    static AclRuntime& instance();

    bool acquire();
    void release();

    // ACL contexts are thread-local: every thread that submits work must bind first.
    void bind_to_thread();
    void* context() const { return ctx_; }
    // aclmdlExecute is not thread-safe; callers serialize on this mutex.
    std::mutex& exec_mtx() { return exec_mtx_; }

    AclRuntime(const AclRuntime&) = delete;
    AclRuntime& operator=(const AclRuntime&) = delete;

private:
    AclRuntime() = default;
    ~AclRuntime();

    std::mutex life_mtx_;
    std::mutex exec_mtx_;
    int refs_ = 0;
    void* ctx_ = nullptr;
};

}  // namespace sm
