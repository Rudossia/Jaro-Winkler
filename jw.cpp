#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>   

char tolower_ascii(char c) {
    if (c >= 'A' && c <= 'Z') return char(c - 'A' + 'a');
    return c;
}

struct JWOptions {
    bool case_insensitive = false;
    double p = 0.1;
    size_t max_prefix = 8;
};

double jaro_similarity(std::string_view s1, std::string_view s2, const JWOptions& opt = {}) {
    size_t n1 = s1.size(), n2 = s2.size();
    if (n1 == 0 && n2 == 0) return 1;
    if (n1 == 0 || n2 == 0) return 0;

    auto eq = [&](char a, char b) { return opt.case_insensitive ? tolower_ascii(a) == tolower_ascii(b) : a == b;};
    int range = int(std::max(n1, n2) / 2) - 1;
    if (range < 0) range = 0;
    std::vector<bool> matched1(n1, false), matched2(n2, false);
    size_t matches = 0;

    for (size_t i = 0; i < n1; ++i) {
        int start = std::max(0, int(i) - range);
        int end   = std::min(int(i) + range + 1, int(n2));
        for (int j = start; j < end; ++j) {
            if (!matched2[j] && eq(s1[i], s2[j])) {
                matched1[i] = matched2[j] = true;
                ++matches;
                break;
            }
        }
    }
    if (matches == 0) return 0;

    std::string m1; m1.reserve(matches);
    std::string m2; m2.reserve(matches);
    for (size_t i = 0; i < n1; ++i) if (matched1[i]) m1.push_back(opt.case_insensitive ? tolower_ascii(s1[i]) : s1[i]);
    for (size_t j = 0; j < n2; ++j) if (matched2[j]) m2.push_back(opt.case_insensitive ? tolower_ascii(s2[j]) : s2[j]);

    size_t transpositions2 = 0;
    for (size_t k = 0; k < matches; ++k)
        if (m1[k] != m2[k]) ++transpositions2;

    double t = transpositions2 / 2.0;
    return ( (double)matches / (double)n1 + (double)matches / (double)n2 + ((double)matches - t) / (double)matches ) / 3.0;
}

double jaro_winkler_similarity(std::string_view s1, std::string_view s2, const JWOptions& opt = {}) {
    double J = jaro_similarity(s1, s2, opt);
    if (J == 0.0) return 0.0;
    size_t prefix = 0;
    size_t bound = std::min({ s1.size(), s2.size(), opt.max_prefix });
    for (; prefix < bound; ++prefix) {
        char a = opt.case_insensitive ? tolower_ascii(s1[prefix]) : s1[prefix];
        char b = opt.case_insensitive ? tolower_ascii(s2[prefix]) : s2[prefix];
        if (a != b) break;
    }
    return J + prefix * opt.p * (1.0 - J);
}

std::vector<std::pair<std::string,std::string>> load_dataset(const std::string& filename) {
    std::vector<std::pair<std::string,std::string>> pairs;
    std::ifstream fin(filename);
    std::string line;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        size_t pos_pipe = line.find('|');
        size_t delim_pos = (pos_pipe != std::string::npos) ? pos_pipe : line.find(',');
        if (delim_pos == std::string::npos) continue;
        std::string left  = line.substr(0, delim_pos);
        std::string right = line.substr(delim_pos + 1);
        auto trim = [](std::string& s) {
            while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
            while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
        };
        trim(left);
        trim(right);
        pairs.emplace_back(left, right);
    }
    return pairs;
}
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_file> [num_runs] [limit]\n";
        std::cerr << "  dataset_file – CSV с двумя строками на строку\n";
        std::cerr << "  num_runs – число повторов (мин. - 3)\n";
        std::cerr << "  limit – сколько пар брать из начала (макс – вест файл)\n";
        return 1;
    }
    std::string filename = argv[1];
    int num_runs = (argc >= 3) ? std::max(1, std::atoi(argv[2])) : 3;
    size_t limit = (argc >= 4) ? std::max(0, std::atoi(argv[3])) : 0;

    auto data = load_dataset(filename);

    size_t n = data.size();
    if (limit > 0 && limit < n) n = limit;
    JWOptions opt;
    opt.case_insensitive = false;

    std::cout << "Processing " << n << " pairs, runs = " << num_runs << "\n";
    double total_sec = 0.0;
    for (int r = 0; r < num_runs; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        double ssum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            ssum += jaro_winkler_similarity(data[i].first, data[i].second, opt);
        }
        auto end = std::chrono::high_resolution_clock::now();
        total_sec += std::chrono::duration<double>(end - start).count();

        if (r == 0) std::cout << "First run sum: " << ssum << "\n";
    }
    double avg_sec = total_sec / num_runs;
    std::cout << "Average time: " << (avg_sec * 1000.0) << " ms\n";
    return 0;
}
