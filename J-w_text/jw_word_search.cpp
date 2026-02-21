#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cctype>

static inline char tolower_ascii(char c) {
    if (c >= 'A' && c <= 'Z') return char(c - 'A' + 'a');
    return c;
}

struct JWOptions {
    bool case_insensitive = false; 
    double p = 0.1; 
    size_t max_prefix = 8;  
};


static double jaro_similarity(std::string_view s1, std::string_view s2,const JWOptions &opt = {}) {
    const size_t n1 = s1.size();
    const size_t n2 = s2.size();
    if (n1 == 0 && n2 == 0) return 1.0;
    if (n1 == 0 || n2 == 0) return 0.0;

    int range = static_cast<int>(std::max(n1, n2) / 2) - 1;
    if (range < 0) range = 0;

    std::vector<bool> matched1(n1, false);
    std::vector<bool> matched2(n2, false);
    size_t matches = 0;

    auto eq = [&](char a, char b) {return opt.case_insensitive ? tolower_ascii(a) == tolower_ascii(b) : a == b;};
    for (size_t i = 0; i < n1; ++i) {
        int start = std::max(0, static_cast<int>(i) - range);
        int end = std::min(static_cast<int>(i) + range + 1, static_cast<int>(n2));
        for (int j = start; j < end; ++j) {
            if (!matched2[j] && eq(s1[i], s2[j])) {
                matched1[i] = matched2[j] = true;
                ++matches;
                break;
            }
        }
    }
    if (matches == 0) return 0.0;

    std::string m1;
    m1.reserve(matches);
    std::string m2;
    m2.reserve(matches);
    for (size_t i = 0; i < n1; ++i) {
        if (matched1[i]) {
            m1.push_back(opt.case_insensitive ? tolower_ascii(s1[i]) : s1[i]);
        }
    }
    for (size_t j = 0; j < n2; ++j) {
        if (matched2[j]) {
            m2.push_back(opt.case_insensitive ? tolower_ascii(s2[j]) : s2[j]);
        }
    }

    size_t transpositions2 = 0;
    for (size_t k = 0; k < matches; ++k) {
        if (m1[k] != m2[k]) {
            ++transpositions2;
        }
    }
    double t = transpositions2 / 2.0;

    return ((double)matches / n1 + (double)matches / n2 + ((double)matches - t) / (double)matches) / 3.0;
}


static double jaro_winkler_similarity(std::string_view s1, std::string_view s2, const JWOptions &opt = {}) {
    double j = jaro_similarity(s1, s2, opt);
    if (j == 0.0) return 0.0;
    size_t prefix = 0;
    size_t bound = std::min({s1.size(), s2.size(), opt.max_prefix});
    for (; prefix < bound; ++prefix) {
        char a = opt.case_insensitive ? tolower_ascii(s1[prefix]) : s1[prefix];
        char b = opt.case_insensitive ? tolower_ascii(s2[prefix]) : s2[prefix];
        if (a != b) break;
    }
    return j + prefix * opt.p * (1.0 - j);
}

static std::vector<std::string> load_words(const std::string &filename) {
    std::vector<std::string> words;
    std::ifstream fin(filename);
    std::string token;
    while (fin >> token) {
        size_t start = 0;
        while (start < token.size() && std::ispunct(static_cast<unsigned char>(token[start]))) {
            ++start;
        }
        size_t end = token.size();
        while (end > start && std::ispunct(static_cast<unsigned char>(token[end - 1]))) {
            --end;
        }
        if (end > start) {
            words.emplace_back(token.substr(start, end - start));
        }
    }
    return words;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset_file> <query_word> <threshold> [case_insensitive]" << '\n'
                  << "  dataset_file    – путь к файлу с текстом" << '\n'
                  << "  query_word      – слово, с которым сравниваем" << '\n'
                  << "  threshold       – порог сходства от 0.0 до 1.0 (например 0.995)" << '\n'
                  << "  case_insensitive – опция '1' сравнивать без учета регистра" << '\n';
        return 1;
    }
    std::string dataset_file = argv[1];
    std::string query_word = argv[2];
    double threshold = std::stod(argv[3]);
    bool case_insensitive = (argc >= 5 && std::string(argv[4]) == "1");

    std::vector<std::string> words = load_words(dataset_file);
    if (words.empty()) {
        std::cerr << "No words loaded from dataset." << '\n';
        return 1;
    }

    JWOptions opt;
    opt.case_insensitive = case_insensitive;
    opt.p = 0.1;
    opt.max_prefix = 4;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::string found_word;
    double found_similarity = 0.0;
    for (const auto &w : words) {
        double sim = jaro_winkler_similarity(query_word, w, opt);
        if (sim >= threshold) {
            found_word = w;
            found_similarity = sim;
            break;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    
    if (!found_word.empty()) {
        std::cout << "Query word: " << query_word << '\n';
        std::cout << "Found word: " << found_word << '\n';
        std::cout << "Similarity: " << found_similarity << '\n';
    } else {
        std::cout << "Query word: " << query_word << '\n';
        std::cout << "No word found meeting the text." << '\n';
    }
    std::cout << "Elapsed time: " << elapsed_ms << " ms" << '\n';
    return 0;
}