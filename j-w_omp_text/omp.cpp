#include <algorithm>
#include <atomic>
#include <cctype>
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
    bool   case_insensitive = false;
    double p = 0.1;
    size_t max_prefix = 4;
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

    for (size_t i = 0; i < n1; ++i) if (matched1[i]) {
        m1.push_back(opt.case_insensitive ? tolower_ascii(s1[i]) : s1[i]);
    }
    for (size_t j = 0; j < n2; ++j) if (matched2[j]) {
        m2.push_back(opt.case_insensitive ? tolower_ascii(s2[j]) : s2[j]);
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

static std::vector<std::string> load_words(const std::string& filename) {
    std::vector<std::string> words;
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return words;
    }

    std::string token;
    while (fin >> token) {
        size_t start = 0;
        while (start < token.size() && std::ispunct((unsigned char)token[start])) ++start;

        size_t end = token.size();
        while (end > start && std::ispunct((unsigned char)token[end - 1])) --end;

        if (end > start) words.emplace_back(token.substr(start, end - start));
    }
    return words;
}

static double run_seq_search(const std::vector<std::string>& words, size_t n,
                             std::string_view query, double threshold,
                             int runs, const JWOptions& opt,
                             long long* first_found_idx_out) {
    double total = 0.0;

    for (int r = 0; r < runs; ++r) {
        long long found = -1;
        double t0 = omp_get_wtime();

        for (size_t i = 0; i < n; ++i) {
            double sim = jaro_winkler_similarity(query, words[i], opt);
            if (sim >= threshold) { found = (long long)i; break; }
        }

        double t1 = omp_get_wtime();
        total += (t1 - t0);

        if (r == 0 && first_found_idx_out) *first_found_idx_out = found;
    }

    return total / double(runs);
}

// Параллельный поиск "как можно ближе к break":
static double run_omp_search(const std::vector<std::string>& words, size_t n,
                             std::string_view query, double threshold,
                             int runs, const JWOptions& opt,
                             int threads, omp_sched_t sched, int chunk,
                             long long* first_found_idx_out) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    omp_set_schedule(sched, chunk);

    double total = 0.0;

    for (int r = 0; r < runs; ++r) {
        std::atomic<long long> best_idx((long long)n);

        double t0 = omp_get_wtime();

        #pragma omp parallel for schedule(runtime)
        for (long long i = 0; i < (long long)n; ++i) {
            long long cur_best = best_idx.load(std::memory_order_relaxed);
            if (i >= cur_best) continue; 
            double sim = jaro_winkler_similarity(query, words[(size_t)i], opt);
            if (sim >= threshold) {
                long long prev = cur_best;
                while (i < prev && !best_idx.compare_exchange_weak(
                           prev, i, std::memory_order_relaxed, std::memory_order_relaxed)) {
                }
            }
        }

        double t1 = omp_get_wtime();
        total += (t1 - t0);

        if (r == 0 && first_found_idx_out) {
            long long bi = best_idx.load(std::memory_order_relaxed);
            *first_found_idx_out = (bi < (long long)n) ? bi : -1;
        }
    }

    return total / double(runs);
}

static void print_found(const std::vector<std::string>& words, size_t n,
                        std::string_view query, double threshold,
                        const JWOptions& opt, long long found_idx) {
    if (found_idx < 0 || (size_t)found_idx >= n) {
        std::cout << "No word found meeting the threshold.\n";
        return;
    }
    double sim = jaro_winkler_similarity(query, words[(size_t)found_idx], opt);
    std::cout << "Found word: " << words[(size_t)found_idx]
              << " | similarity=" << sim
              << " | index=" << found_idx << "\n";
}


struct Args {
    std::string file;
    std::string query;
    double threshold = 0.0;

    int runs = 10;
    long long limit = 0;

    int tmin = 2;
    int tmax = 10;

    std::string schedule = "guided"; 
    int chunk = 64;

    bool case_insensitive = false;
    double p = 0.1;
    size_t max_prefix = 4;

    std::string out_dat; 
};

static void usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " <text_file> <query_word> <threshold> [options]\n\n"
        << "Options:\n"
        << "  --runs N            (default 10, min 3)\n"
        << "  --limit N           (default 0 = all)\n"
        << "  --tmin N            (default 2)\n"
        << "  --tmax N            (default 10)\n"
        << "  --schedule NAME     guided|dynamic|static (default guided)\n"
        << "  --chunk N           (default 64)\n"
        << "  --ci                case-insensitive\n"
        << "  --p X               Winkler p (default 0.1)\n"
        << "  --prefix N          max_prefix (default 4)\n"
        << "  --out FILE.dat      write results to .dat (threads avg_ms speedup efficiency)\n";
}

static bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 4) return false;

    a.file = argv[1];
    a.query = argv[2];
    a.threshold = std::stod(argv[3]);

    for (int i = 4; i < argc; ++i) {
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
        } else if (k == "--ci") {
            a.case_insensitive = true;
        } else if (k == "--p") {
            a.p = std::atof(need_val("--p"));
        } else if (k == "--prefix") {
            a.max_prefix = (size_t)std::max(1, std::atoi(need_val("--prefix")));
        } else if (k == "--out") {
            a.out_dat = need_val("--out");
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

    auto words = load_words(args.file);
    if (words.empty()) {
        std::cerr << "No words loaded from dataset.\n";
        return 2;
    }

    size_t n = words.size();
    if (args.limit > 0 && (size_t)args.limit < n) n = (size_t)args.limit;

    JWOptions opt;
    opt.case_insensitive = args.case_insensitive;
    opt.p = args.p;
    opt.max_prefix = args.max_prefix;

    std::cout << "Words: " << n << " (loaded " << words.size() << "), runs=" << args.runs << "\n";
    std::cout << "Query: " << args.query
              << " | threshold=" << args.threshold
              << " | ci=" << (args.case_insensitive ? "on" : "off")
              << " | p=" << opt.p
              << " | prefix=" << opt.max_prefix << "\n\n";

    long long seq_found = -1;
    double T1 = run_seq_search(words, n, args.query, args.threshold, args.runs, opt, &seq_found);

    std::cout << "sequential avg time: " << (T1 * 1000.0) << " ms\n";
    print_found(words, n, args.query, args.threshold, opt, seq_found);
    std::cout << "\n";

    omp_sched_t sched = parse_sched(args.schedule);

    std::cout << "threads| avg_ms | speedup | efficiency\n";

    std::ofstream dat;
    bool write_dat = !args.out_dat.empty();
    if (write_dat) {
        dat.open(args.out_dat);
        if (!dat) {
            std::cerr << "Failed to open --out file: " << args.out_dat << "\n";
            write_dat = false;
        } else {
            dat << "# threads avg_ms speedup efficiency\n";
        }
    }

    for (int th = args.tmin; th <= args.tmax; ++th) {
        if (th == 1) continue;

        long long par_found = -1;
        double Tp = run_omp_search(words, n, args.query, args.threshold, args.runs, opt,
                                   th, sched, args.chunk, &par_found);

        double S = T1 / Tp;
        double E = S / double(th);

        std::cout << th << " | " << (Tp * 1000.0) << " | " << S << " | " << E << "\n";

        if (write_dat) {
            dat << th << " " << (Tp * 1000.0) << " " << S << " " << E << "\n";
        }
    }

    if (write_dat) {
        dat.close();
        std::cout << "\nWrote: " << args.out_dat << "\n";
    }

    return 0;
}

