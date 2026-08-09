#ifndef TATAMI_MULT_MULTIPLE_VECTORS_UTILS_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_UTILS_HPP

#include <optional>
#include <vector>

namespace tatami_mult {

template<typename Container_>
class LiberateArraysScope {
public:
    LiberateArraysScope(Container_& x) : my_x(x) {}

    ~LiberateArraysScope() {
        liberate(my_x);
    }
private:
    Container_& my_x;

    template<typename Value_>
    void liberate(std::optional<Value_>& x) {
        if (x.has_value()) {
            liberate(*x);
        }
    }

    template<typename Value_>
    void liberate(std::vector<Value_>& x) {
        for (auto& y : x) {
            liberate(y);
        }
    }

    template<typename Value_>
    void liberate(const Value_* x) {
        if (x) {
            delete [] x;
        }
    }
};

}

#endif
