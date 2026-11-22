#include "sm/acl_runtime.hpp"

#include <acl/acl.h>

#include <cstdio>

namespace sm {

namespace {
thread_local void* tls_ctx = nullptr;
}  // namespace

AclRuntime& AclRuntime::instance() {
    static AclRuntime rt;
    return rt;
}

AclRuntime::~AclRuntime() {
    if (refs_ <= 0) return;
    if (ctx_) aclrtDestroyContext(static_cast<aclrtContext>(ctx_));
    aclrtResetDevice(0);
    aclFinalize();
}

bool AclRuntime::acquire() {
    std::lock_guard<std::mutex> lk(life_mtx_);
    if (refs_ > 0) {
        refs_++;
        return true;
    }
    if (aclInit(nullptr) != ACL_SUCCESS) {
        fprintf(stderr, "[acl] aclInit failed\n");
        return false;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        fprintf(stderr, "[acl] setDevice failed\n");
        aclFinalize();
        return false;
    }
    aclrtContext ctx = nullptr;
    if (aclrtCreateContext(&ctx, 0) != ACL_SUCCESS) {
        fprintf(stderr, "[acl] createContext failed\n");
        aclrtResetDevice(0);
        aclFinalize();
        return false;
    }
    ctx_ = ctx;
    tls_ctx = ctx;
    refs_ = 1;
    return true;
}

void AclRuntime::release() {
    std::lock_guard<std::mutex> lk(life_mtx_);
    if (refs_ <= 0) return;
    if (--refs_ > 0) return;
    if (ctx_) {
        aclrtDestroyContext(static_cast<aclrtContext>(ctx_));
        ctx_ = nullptr;
    }
    tls_ctx = nullptr;
    aclrtResetDevice(0);
    aclFinalize();
}

void AclRuntime::bind_to_thread() {
    if (ctx_ && tls_ctx != ctx_) {
        aclrtSetCurrentContext(static_cast<aclrtContext>(ctx_));
        tls_ctx = ctx_;
    }
}

}  // namespace sm
