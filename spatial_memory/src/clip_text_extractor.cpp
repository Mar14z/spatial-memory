#include "sm/clip_extractor.hpp"

#include <acl/acl.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace sm {

ClipTextExtractor::~ClipTextExtractor() {
    if (in_ds_) aclmdlDestroyDataset(static_cast<aclmdlDataset*>(in_ds_));
    if (out_ds_) aclmdlDestroyDataset(static_cast<aclmdlDataset*>(out_ds_));
    if (in_dev_) aclrtFree(in_dev_);
    if (out_dev_) aclrtFree(out_dev_);
    if (model_id_) aclmdlUnload(model_id_);
    if (ctx_) {
        aclrtDestroyContext(static_cast<aclrtContext>(ctx_));
        aclrtResetDevice(0);
    }
    if (ready_) aclFinalize();
}

bool ClipTextExtractor::init(const std::string& om_path) {
    if (aclInit(nullptr) != ACL_SUCCESS) {
        fprintf(stderr, "[clip-text] aclInit failed\n");
        return false;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) return false;
    aclrtContext ctx = nullptr;
    if (aclrtCreateContext(&ctx, 0) != ACL_SUCCESS) return false;
    ctx_ = ctx;

    if (aclmdlLoadFromFile(om_path.c_str(), &model_id_) != ACL_SUCCESS) {
        fprintf(stderr, "[clip-text] load model failed: %s\n", om_path.c_str());
        return false;
    }
    aclmdlDesc* desc = aclmdlCreateDesc();
    aclmdlGetDesc(desc, model_id_);
    in_bytes_ = aclmdlGetInputSizeByIndex(desc, 0);   // 77 * 8 (int64)
    out_bytes_ = aclmdlGetOutputSizeByIndex(desc, 0); // 512 * 4
    aclmdlDestroyDesc(desc);
    if (in_bytes_ != (size_t)77 * sizeof(int64_t) || out_bytes_ == 0) {
        fprintf(stderr, "[clip-text] unexpected IO sizes in=%zu out=%zu\n", in_bytes_, out_bytes_);
        return false;
    }

    if (aclrtMalloc(&in_dev_, in_bytes_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return false;
    if (aclrtMalloc(&out_dev_, out_bytes_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return false;

    aclmdlDataset* in_ds = aclmdlCreateDataset();
    aclmdlDataset* out_ds = aclmdlCreateDataset();
    if (aclmdlAddDatasetBuffer(in_ds, aclCreateDataBuffer(in_dev_, in_bytes_)) != ACL_SUCCESS) return false;
    if (aclmdlAddDatasetBuffer(out_ds, aclCreateDataBuffer(out_dev_, out_bytes_)) != ACL_SUCCESS) return false;
    in_ds_ = in_ds;
    out_ds_ = out_ds;
    ready_ = true;
    return true;
}

bool ClipTextExtractor::encode(const int64_t* ids, size_t n, std::vector<float>& out) {
    if (!ready_ || !ids || n > 77) return false;
    int64_t buf[77];
    for (size_t i = 0; i < 77; i++) buf[i] = (i < n) ? ids[i] : 49407;

    if (aclrtMemcpy(in_dev_, in_bytes_, buf, in_bytes_,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return false;
    }

    auto t0 = std::chrono::steady_clock::now();
    if (aclmdlExecute(model_id_, static_cast<aclmdlDataset*>(in_ds_),
                      static_cast<aclmdlDataset*>(out_ds_)) != ACL_SUCCESS) {
        return false;
    }
    auto t1 = std::chrono::steady_clock::now();
    last_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const size_t dim = out_bytes_ / sizeof(float);
    out.resize(dim);
    if (aclrtMemcpy(out.data(), out_bytes_, out_dev_, out_bytes_,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    double s = 0.0;
    for (float x : out) s += (double)x * (double)x;
    if (s > 0.0) {
        float inv = (float)(1.0 / std::sqrt(s));
        for (float& x : out) x *= inv;
    }
    return true;
}

}  // namespace sm
