
#include <Interpolation.hpp>
#include <chrono>
#include <cmath>
#include <iostream>

double f(std::vector<double> point) {
  double prod = 1.0;
  for (size_t dim = 0; dim < point.size(); ++dim) {
    prod *= point[dim];
  }
  return prod;
}

class GoldenPointDistribution {
  static constexpr double golden_ratio = 0.5 * (1.0 + sqrt(5));

 public:
  double operator()(size_t idx) {
    double value = (idx + 1) * golden_ratio;
    return value - int(value);
  }
};

class SimplePointDistribution {
 public:
  double operator()(size_t idx) {
    if (idx == 0) {
      return 0.0;
    } else if (idx == 1) {
      return 1.0;
    } else {
      return 0.5;
    }
  }
};

class MonomialFunctions {
 public:
  std::function<double(double)> operator()(size_t idx) {
    return [=](double x) { return pow(x, idx); };
  }
};

template <class T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "[";
  for (auto ii = v.begin(); ii != v.end(); ++ii) {
    os << " " << *ii;
  }
  os << " ]";
  return os;
}

// using namespace fsi;
int main() {
  size_t d = 12;
  size_t bound = 16;
  fsi::BoundedSumIterator it(d, bound);
  std::vector<MonomialFunctions> phi(d);
  std::vector<GoldenPointDistribution> x(d);
  auto start = std::chrono::high_resolution_clock::now();
  auto result = fsi::interpolate(f, it, phi, x);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";

  // for (size_t i = 0; i < result.data.size(); ++i) {
  //   std::cout << result.data[i] << "\n\n";
  // }
}
