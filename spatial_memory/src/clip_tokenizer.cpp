#include "sm/clip_tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <limits>

namespace sm {

namespace {

std::string utf8_encode(int cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

const std::vector<std::string>& byte_to_unicode() {
    static std::vector<std::string> table;
    if (!table.empty()) return table;
    table.resize(256);
    std::vector<int> bs;
    for (int b = 33; b <= 126; b++) bs.push_back(b);
    for (int b = 161; b <= 172; b++) bs.push_back(b);
    for (int b = 174; b <= 255; b++) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool in = false;
        for (int x : bs) {
            if (x == b) {
                in = true;
                break;
            }
        }
        if (!in) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    for (size_t i = 0; i < bs.size(); i++) table[bs[i]] = utf8_encode(cs[i]);
    return table;
}

std::vector<std::string> split_utf8(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        }
        if (i + len > s.size()) len = 1;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

bool is_alpha(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

}  // namespace

bool ClipTokenizer::load(const std::string& vocab_tsv, const std::string& merges_txt) {
    std::ifstream vf(vocab_tsv);
    if (!vf) return false;
    std::string line;
    while (std::getline(vf, line)) {
        if (line.empty()) continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        int64_t id = std::stoll(line.substr(0, tab));
        const std::string hex = line.substr(tab + 1);
        std::string tok;
        tok.reserve(hex.size() / 2);
        for (size_t i = 0; i + 1 < hex.size(); i += 2) {
            tok += (char)std::stoi(hex.substr(i, 2), nullptr, 16);
        }
        tok_to_id_[tok] = id;
        if ((int64_t)id_to_tok_.size() <= id) id_to_tok_.resize((size_t)id + 1);
        id_to_tok_[(size_t)id] = tok;
    }

    std::ifstream mf(merges_txt);
    if (!mf) return false;
    int rank = 0;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        pair_rank_[line] = rank++;
    }
    loaded_ = !tok_to_id_.empty() && !pair_rank_.empty();
    return loaded_;
}

std::vector<int64_t> ClipTokenizer::encode(const std::string& text, size_t max_len) const {
    std::vector<int64_t> ids;
    ids.push_back(49406);  // <|startoftext|>

    // normalize: lowercase + collapse whitespace
    std::string norm;
    bool pending_space = false;
    for (unsigned char c : text) {
        if (is_space(c)) {
            pending_space = true;
            continue;
        }
        if (pending_space && !norm.empty()) norm += ' ';
        pending_space = false;
        norm += (char)((c >= 'A' && c <= 'Z') ? c + 32 : c);
    }

    const auto& b2u = byte_to_unicode();
    size_t i = 0;
    while (i < norm.size() && ids.size() + 1 < max_len) {
        unsigned char c = (unsigned char)norm[i];
        if (is_space(c)) {
            i++;
            continue;
        }
        size_t start = i;
        if (is_alpha(c)) {
            while (i < norm.size() && is_alpha((unsigned char)norm[i])) i++;
        } else if (is_digit(c)) {
            i++;
        } else {
            while (i < norm.size() && !is_space((unsigned char)norm[i]) &&
                   !is_alpha((unsigned char)norm[i]) && !is_digit((unsigned char)norm[i])) {
                i++;
            }
        }
        std::string word = norm.substr(start, i - start);

        // CLIP tokenizer: byte-level chars, no space prefix; the LAST char carries
        // the end-of-word marker "</w>" (e.g. "photo" -> p,h,o,t,o</w> -> "photo</w>").
        std::string mapped;
        for (unsigned char b : word) mapped += b2u[b];

        std::vector<std::string> chars = split_utf8(mapped);
        if (chars.empty()) continue;
        chars.back() += "</w>";
        while (chars.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_i = chars.size();
            for (size_t k = 0; k + 1 < chars.size(); k++) {
                std::string pair = chars[k] + " " + chars[k + 1];
                auto it = pair_rank_.find(pair);
                if (it != pair_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = k;
                }
            }
            if (best_i == chars.size()) break;
            chars[best_i] += chars[best_i + 1];
            chars.erase(chars.begin() + (ptrdiff_t)(best_i + 1));
        }

        for (const auto& ch : chars) {
            auto it = tok_to_id_.find(ch);
            if (it != tok_to_id_.end() && ids.size() + 1 < max_len) ids.push_back(it->second);
        }
    }

    ids.push_back(49407);  // <|endoftext|>
    while (ids.size() < max_len) ids.push_back(49407);
    if (ids.size() > max_len) ids.resize(max_len);
    return ids;
}

}  // namespace sm
