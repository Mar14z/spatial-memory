#include "sm/clip_extractor.hpp"

#include <acl/acl.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "sm/acl_runtime.hpp"

namespace sm {

namespace {
// CLIP ViT-B/32 preprocessing constants
constexpr float kMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float kStd[3] = {0.26862954f, 0.26130258f, 0.27577711f};
constexpr int kSize = 224;
}  // namespace

ClipVisionExtractor::~ClipVisionExtractor() {
    if (in_ds_) aclmdlDestroyDataset(static_cast<aclmdlDataset*>(in_ds_));
    if (out_ds_) aclmdlDestroyDataset(static_cast<aclmdlDataset*>(out_ds_));
    if (in_dev_) aclrtFree(in_dev_);
    if (out_dev_) aclrtFree(out_dev_);
    if (model_id_) aclmdlUnload(model_id_);
    if (holds_runtime_) AclRuntime::instance().release();
}

bool ClipVisionExtractor::init(const std::string& om_path) {
    auto& rt = AclRuntime::instance();
    if (!rt.acquire()) {
        fprintf(stderr, "[clip] acl runtime unavailable\n");
        return false;
    }
    holds_runtime_ = true;
    rt.bind_to_thread();
    auto fail = [this]() {
        if (holds_runtime_) {
            AclRuntime::instance().release();
            holds_runtime_ = false;
        }
        return false;
    };

    if (aclmdlLoadFromFile(om_path.c_str(), &model_id_) != ACL_SUCCESS) {
        fprintf(stderr, "[clip] load model failed: %s\n", om_path.c_str());
        return fail();
    }
    aclmdlDesc* desc = aclmdlCreateDesc();
    aclmdlGetDesc(desc, model_id_);
    in_bytes_ = aclmdlGetInputSizeByIndex(desc, 0);
    out_bytes_ = aclmdlGetOutputSizeByIndex(desc, 0);
    aclmdlDestroyDesc(desc);
    if (in_bytes_ != (size_t)kSize * kSize * 3 * sizeof(float) || out_bytes_ == 0) {
        fprintf(stderr, "[clip] unexpected IO sizes in=%zu out=%zu\n", in_bytes_, out_bytes_);
        return fail();
    }

    if (aclrtMalloc(&in_dev_, in_bytes_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return fail();
    if (aclrtMalloc(&out_dev_, out_bytes_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return fail();

    aclmdlDataset* in_ds = aclmdlCreateDataset();
    aclmdlDataset* out_ds = aclmdlCreateDataset();
    if (aclmdlAddDatasetBuffer(in_ds, aclCreateDataBuffer(in_dev_, in_bytes_)) != ACL_SUCCESS) {
        aclmdlDestroyDataset(in_ds);
        aclmdlDestroyDataset(out_ds);
        return fail();
    }
    if (aclmdlAddDatasetBuffer(out_ds, aclCreateDataBuffer(out_dev_, out_bytes_)) != ACL_SUCCESS) {
        aclmdlDestroyDataset(in_ds);
        aclmdlDestroyDataset(out_ds);
        return fail();
    }
    in_ds_ = in_ds;
    out_ds_ = out_ds;
    ready_ = true;
    return true;
}

bool ClipVisionExtractor::extract(const uint8_t* rgb224, std::vector<float>& out) {
    if (!ready_ || !rgb224) return false;
    auto& rt = AclRuntime::instance();
    rt.bind_to_thread();
    std::lock_guard<std::mutex> lk(rt.exec_mtx());
    const size_t n = (size_t)kSize * kSize;

    // HWC uint8 -> CHW float, (x/255 - mean)/std
    std::vector<float> host(n * 3);
    for (int c = 0; c < 3; c++) {
        float scale = 1.0f / (255.0f * kStd[c]);
        float bias = -kMean[c] / kStd[c];
        float* dst = host.data() + (size_t)c * n;
        const uint8_t* src = rgb224 + c;
        for (size_t i = 0; i < n; i++, src += 3) {
            dst[i] = (float)(*src) * scale + bias;
        }
    }

    if (aclrtMemcpy(in_dev_, in_bytes_, host.data(), in_bytes_,
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
