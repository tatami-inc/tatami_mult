#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <vector>

inline std::vector<double> simulate_strided_sparse_matrix(const int primary, const int secondary, const int stride, unsigned long long seed) {
    std::vector<double> output;
    std::mt19937_64 rng(seed);
    std::normal_distribution<> dist;

    if (stride != 0) {
        output.reserve(primary * secondary);
        for (int p = 0; p < primary; ++p) {
            if (p % stride == 1) {
                for (int s = 0; s < secondary; ++s) {
                    if (s % stride == 1) {
                        output.push_back(dist(rng));
                    } else {
                        output.push_back(0);
                    }
                }
            } else {
                output.insert(output.end(), secondary, 0);
            }
        }
    } else {
        output.resize(primary * secondary);
    }

    return output;
}

#endif
