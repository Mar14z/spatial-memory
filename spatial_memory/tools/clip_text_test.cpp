// Verify CLIP text encoder end to end: tokenize -> NPU -> 512-d feature -> cosine matrix.
#include <cstdio>
#include <vector>

#include "sm/clip_extractor.hpp"
#include "sm/clip_tokenizer.hpp"

int main(int argc, char** argv) {
    const char* om = (argc > 1) ? argv[1] : "/root/clip_text_om.om";
    const char* vocab = (argc > 2) ? argv[2] : "/root/clip_data/vocab.tsv";
    const char* merges = (argc > 3) ? argv[3] : "/root/clip_data/merges.txt";

    sm::ClipTokenizer tk;
    if (!tk.load(vocab, merges)) {
        printf("TOKENIZER_LOAD_FAIL\n");
        return 1;
    }
    sm::ClipTextExtractor ex;
    if (!ex.init(om)) {
        printf("INIT_FAIL\n");
        return 1;
    }

    const char* texts[] = {
        "a photo of a cup",
        "a red mug",
        "a blue chair",
        "a car",
    };
    const int N = 4;
    std::vector<std::vector<float>> feats(N);

    for (int i = 0; i < N; i++) {
        auto ids = tk.encode(texts[i]);
        printf("[%d] \"%s\" -> ", i, texts[i]);
        for (size_t k = 0; k < ids.size(); k++) {
            if (ids[k] == 49407) break;
            printf("%lld ", (long long)ids[k]);
        }
        printf("| ");
        if (!ex.encode(ids.data(), ids.size(), feats[i])) {
            printf("ENCODE_FAIL\n");
            return 1;
        }
        printf("ms=%.2f\n", ex.last_ms());
    }

    printf("\ncosine matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.f;
            for (size_t k = 0; k < feats[i].size(); k++) dot += feats[i][k] * feats[j][k];
            printf("%7.3f", dot);
        }
        printf("   %s\n", texts[i]);
    }
    printf("\nexpect: cup/mug highest (same semantics), car lowest\n");
    return 0;
}
