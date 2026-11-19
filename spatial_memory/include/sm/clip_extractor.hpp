#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sm {

// CLIP ViT-B/32 vision encoder via Ascend ACL.
// Input: 224x224 RGB uint8. Output: 512-d L2-normalized feature.
class ClipVisionExtractor {
public:
    ClipVisionExtractor() = default;
    ~ClipVisionExtractor();
    ClipVisionExtractor(const ClipVisionExtractor&) = delete;
    ClipVisionExtractor& operator=(const ClipVisionExtractor&) = delete;

    bool init(const std::string& om_path);
    bool ready() const { return ready_; }
    bool extract(const uint8_t* rgb224, std::vector<float>& out);
    double last_ms() const { return last_ms_; }

private:
    void* ctx_ = nullptr;
    uint32_t model_id_ = 0;
    void* in_dev_ = nullptr;
    void* out_dev_ = nullptr;
    void* in_ds_ = nullptr;
    void* out_ds_ = nullptr;
    size_t in_bytes_ = 0;
    size_t out_bytes_ = 0;
    bool ready_ = false;
    double last_ms_ = 0.0;
};

}  // namespace sm
