// Verify the CLIP vision tower end to end (synthetic images, no camera needed).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "sm/clip_extractor.hpp"

int main(int argc, char** argv) {
    const char* om = (argc > 1) ? argv[1] : "/root/clip_vision_om.om";
    sm::ClipVisionExtractor ex;
    if (!ex.init(om)) {
        printf("INIT_FAIL\n");
        return 1;
    }
    printf("init ok: %s\n", om);

    std::vector<uint8_t> a(224 * 224 * 3), b(224 * 224 * 3);
    for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
            int i = (y * 224 + x) * 3;
            a[i] = 220;              // warm red-ish gradient
            a[i + 1] = (uint8_t)(y / 2);
            a[i + 2] = (uint8_t)(x / 2);
            b[i] = (uint8_t)(x / 2); // cool blue-ish gradient
            b[i + 1] = (uint8_t)(y / 2);
            b[i + 2] = 220;
        }
    }

    std::vector<float> fa, fb;
    if (!ex.extract(a.data(), fa)) { printf("EXTRACT_A_FAIL\n"); return 1; }
    printf("A: dim=%zu ms=%.2f norm=%.3f\n", fa.size(), ex.last_ms(),
           std::sqrt([&] { double s = 0; for (float v : fa) s += v * v; return s; }()));
    if (!ex.extract(b.data(), fb)) { printf("EXTRACT_B_FAIL\n"); return 1; }
    printf("B: dim=%zu ms=%.2f\n", fb.size(), ex.last_ms());

    float dot = 0.f;
    for (size_t i = 0; i < fa.size(); i++) dot += fa[i] * fb[i];
    printf("cosine(A,B)=%.4f (different images, expect < 0.99)\n", dot);

    const int N = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        if (!ex.extract(a.data(), fa)) { printf("EXTRACT_LOOP_FAIL\n"); return 1; }
    }
    auto t1 = std::chrono::steady_clock::now();
    printf("RESULT avg=%.2f ms over %d runs\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count() / N, N);
    return 0;
}
