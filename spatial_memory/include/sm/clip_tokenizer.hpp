#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sm {

// Minimal CLIP (ViT-B/32) BPE tokenizer, GPT-2 byte-level variant.
// Loads vocab.tsv (id<TAB>hex(utf8)) and merges.txt; emits int64 ids padded to 77.
class ClipTokenizer {
public:
    bool load(const std::string& vocab_tsv, const std::string& merges_txt);
    bool loaded() const { return loaded_; }
    std::vector<int64_t> encode(const std::string& text, size_t max_len = 77) const;

private:
    std::vector<std::string> id_to_tok_;
    std::unordered_map<std::string, int64_t> tok_to_id_;
    std::unordered_map<std::string, int> pair_rank_;
    bool loaded_ = false;
};

}  // namespace sm
