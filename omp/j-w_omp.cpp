#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <omp.h>

static inline char tolower_ascii(char c) {
    return (c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : c;
}

struct JWOptions {
    bool case_insensitive = false;
    double p = 0.1;
    size_t max_prefix = 8;
};


static double jaro_similarity(std::string_view s1, std::string_view s2, const JWOptions& opt) {
    const size_t n1 = s1.size();
    const size_t n2 = s2.size();
    if (n1 == 0 && n2 == 0) return 1.0;
    if (n1 == 0 || n2 == 0) return 0.0;

    auto eq = [&](char a, char b) {
        return opt.case_insensitive ? (tolower_ascii(a) == tolower_ascii(b)) : (a == b);
    };

    int range = int(std::max(n1, n2) / 2) - 1;
    if (range < 0) range = 0;

    std::vector<bool> matched1(n1, false);
    std::vector<bool> matched2(n2, false);

    size_t matches = 0;

    for (size_t i = 0; i < n1; ++i) {
        int start = std::max(0, int(i) - range);
        int end   = std::min(int(i) + range + 1, int(n2));
        for (int j = start; j < end; ++j) {
            if (!matched2[(size_t)j] && eq(s1[i], s2[(size_t)j])) {
                matched1[i] = true;
                matched2[(size_t)j] = true;
                ++matches;
                break;
            }
        }
    }

    if (matches == 0) return 0.0;

    std::string m1; m1.reserve(matches);
    std::string m2; m2.reserve(matches);

    for (size_t i = 0; i < n1; ++i) {
        if (matched1[i]) {
            char c = opt.case_insensitive ? tolower_ascii(s1[i]) : s1[i];
            m1.push_back(c);
        }
    }
    for (size_t j = 0; j < n2; ++j) {
        if (matched2[j]) {
            char c = opt.case_insensitive ? tolower_ascii(s2[j]) : s2[j];
            m2.push_back(c);
        }
    }

    size_t transpositions2 = 0;
    for (size_t k = 0; k < matches; ++k) {
        if (m1[k] != m2[k]) ++transpositions2;
    }
    const double t = transpositions2 / 2.0;

    const double dn1 = double(n1), dn2 = double(n2), dm = double(matches);
    return ((dm / dn1) + (dm / dn2) + ((dm - t) / dm)) / 3.0;
}

static double jaro_winkler_similarity(std::string_view s1, std::string_view s2, const JWOptions& opt) {
    const double J = jaro_similarity(s1, s2, opt);
    if (J == 0.0) return 0.0;

    const size_t bound = std::min({ s1.size(), s2.size(), opt.max_prefix });
    size_t prefix = 0;
    for (; prefix < bound; ++prefix) {
        char a = opt.case_insensitive ? tolower_ascii(s1[prefix]) : s1[prefix];
        char b = opt.case_insensitive ? tolower_ascii(s2[prefix]) : s2[prefix];
        if (a != b) break;
    }
    return J + double(prefix) * opt.p * (1.0 - J);
}


static inline void trim_inplace(std::string& s) {
    while (!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && std::isspace((unsigned char)s.back()))  s.pop_back();
}

static std::vector<std::pair<std::string, std::string>> load_dataset(const std::string& filename) {
    std::vector<std::pair<std::string, std::string>> pairs;
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return pairs;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        size_t pos_pipe = line.find('|');
        size_t delim = (pos_pipe != std::string::npos) ? pos_pipe : line.find(',');
        if (delim == std::string::npos) continue;

        std::string left  = line.substr(0, delim);
        std::string right = line.substr(delim + 1);
        trim_inplace(left);
        trim_inplace(right);

        pairs.emplace_back(std::move(left), std::move(right));
    }
    return pairs;
}
// Это "сколько потенциально проверок символов" делает внутренний поиск совпадений.

static uint64_t work_upper_pair(std::string_view s1, std::string_view s2) {
    const size_t n1 = s1.size(), n2 = s2.size();
    if (n1 == 0 || n2 == 0) return 0;

    int range = int(std::max(n1, n2) / 2) - 1;
    if (range < 0) range = 0;

    uint64_t w = 0;
    for (size_t i = 0; i < n1; ++i) {
        int start = std::max(0, int(i) - range);
        int end   = std::min(int(i) + range + 1, int(n2));
        w += uint64_t(end - start);
    }
    return w;
}


static inline double now_sec() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// ЧИСТО последовательный: никаких omp parallel
static double run_seq(const std::vector<std::pair<std::string,std::string>>& data,
                      size_t n, int runs, const JWOptions& opt, double* first_sum_out) {
    double total = 0.0;

    for (int r = 0; r < runs; ++r) {
        double ssum = 0.0;
        double t0 = now_sec();

        for (size_t i = 0; i < n; ++i) {
            ssum += jaro_winkler_similarity(data[i].first, data[i].second, opt);
        }

        double t1 = now_sec();
        total += (t1 - t0);

        if (r == 0 && first_sum_out) *first_sum_out = ssum;
    }
    return total / double(runs);
}

static double run_omp(const std::vector<std::pair<std::string,std::string>>& data,size_t n, int runs, const JWOptions& opt,int threads, omp_sched_t sched, int chunk,double* first_sum_out) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    omp_set_schedule(sched, chunk);

    double total = 0.0;

    for (int r = 0; r < runs; ++r) {
        double ssum = 0.0;
        double t0 = omp_get_wtime();

        #pragma omp parallel for reduction(+:ssum) schedule(runtime)
        for (long long i = 0; i < (long long)n; ++i) {
            const auto& pr = data[(size_t)i];
            ssum += jaro_winkler_similarity(pr.first, pr.second, opt);
        }

        double t1 = omp_get_wtime();
        total += (t1 - t0);

        if (r == 0 && first_sum_out) *first_sum_out = ssum;
    }
    return total / double(runs);
}


struct Args {
    std::string file;
    int runs = 10;
    long long limit = 0;

    int tmin = 2;
    int tmax = 10;

    std::string schedule = "guided";
    int chunk = 64;

    bool print_work = true;
    bool case_insensitive = false;
};

void usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " <dataset_file> [options]\n\n"
        << "Options:\n"
        << "  --runs N            (default 10, min 3)\n"
        << "  --limit N           (default 0 = all)\n"
        << "  --tmin N            (default 2)\n"
        << "  --tmax N            (default 10)\n"
        << "  --schedule NAME     guided|dynamic|static (default guided)\n"
        << "  --chunk N           (default 64)\n"
        << "  --no-work           don't compute W_upper metric\n"
        << "  --ci                case-insensitive сравнение (default off)\n\n"
        << "Example:\n"
        << "  " << prog << " data.csv --runs 10 --tmin 2 --tmax 10 --schedule guided --chunk 64\n";
}

static bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 2) return false;
    a.file = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];

        auto need_val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (k == "--runs") {
            a.runs = std::max(3, std::atoi(need_val("--runs")));
        } else if (k == "--limit") {
            a.limit = std::atoll(need_val("--limit"));
            if (a.limit < 0) a.limit = 0;
        } else if (k == "--tmin") {
            a.tmin = std::max(1, std::atoi(need_val("--tmin")));
        } else if (k == "--tmax") {
            a.tmax = std::max(1, std::atoi(need_val("--tmax")));
        } else if (k == "--schedule") {
            a.schedule = need_val("--schedule");
        } else if (k == "--chunk") {
            a.chunk = std::max(1, std::atoi(need_val("--chunk")));
        } else if (k == "--no-work") {
            a.print_work = false;
        } else if (k == "--ci") {
            a.case_insensitive = true;
        } else if (k == "--help" || k == "-h") {
            return false;
        } else {
            std::cerr << "Unknown option: " << k << "\n";
            return false;
        }
    }

    if (a.tmin > a.tmax) std::swap(a.tmin, a.tmax);
    return true;
}

static omp_sched_t parse_sched(const std::string& s) {
    if (s == "guided")  return omp_sched_guided;
    if (s == "dynamic") return omp_sched_dynamic;
    if (s == "static")  return omp_sched_static;
    return omp_sched_guided;
}


int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 1;
    }

    auto data = load_dataset(args.file);
    if (data.empty()) {
        std::cerr << "Dataset is empty or failed to load.\n";
        return 2;
    }

    size_t n = data.size();
    if (args.limit > 0 && (size_t)args.limit < n) n = (size_t)args.limit;

    JWOptions opt;
    opt.case_insensitive = args.case_insensitive;

    //std::cout << "File: " << args.file << "\n";
    std::cout << "Pairs: " << n << " (loaded " << data.size() << "), runs=" << args.runs << "\n";
    //std::cout << "Threads: " << args.tmin << ".." << args.tmax << " | schedule=" << args.schedule << " chunk=" << args.chunk << "\n";

    /*if (args.print_work) {
        uint64_t W = 0;
        for (size_t i = 0; i < n; ++i) {
            W += work_upper_pair(data[i].first, data[i].second);
        }
        std::cout << "W_upper_total: " << W << "  (avg per pair: " << (double)W / (double)n << ")\n\n";
    }*/

    double seq_sum = 0.0;
    double T1 = run_seq(data, n, args.runs, opt, &seq_sum);
    std::cout << "sequental sum: " << seq_sum << "\n";
    std::cout << "sequental avg time: " << (T1 * 1000.0) << " ms\n\n";

    omp_sched_t sched = parse_sched(args.schedule);

    std::cout << "threads| avg_ms | speedup | efficiency\n";

    for (int th = args.tmin; th <= args.tmax; ++th) {
        if (th == 1) continue; 

        double par_sum = 0.0;
        double Tp = run_omp(data, n, args.runs, opt, th, sched, args.chunk, &par_sum);

        double S = T1 / Tp;
        double E = S / double(th);
        //double delta = par_sum - seq_sum; 

        std::cout << th << " | " << (Tp * 1000.0) << " | " << S << " | " << E <<  "\n";
    }

    return 0;
}
