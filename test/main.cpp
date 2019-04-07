/* Copyright 2019 The fast_sparse_interpolation Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Interpolation.hpp>
#include <Iterators.hpp>
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

std::ostream &operator<<(std::ostream &os, fsi::MultiDimVector const &v) {
  for (size_t i = 0; i < v.data.size(); ++i) {
    std::cout << v.data[i] << "\n\n";
  }

  return os;
}

// TODO: refactoring:
/**
 * - Interface for MultiDimVector
 * - Separate Functions for L- and U-Multiplication, Creation of LU decomposition, computation of
 * function values
 * - Pass a callback function to iterator instead of calling it.next() - this might improve the
 * vector functions (could be made recursive or even loops for fixed dimension implementation)
 * - Typed interface?
 * - Tests?
 * - More point distributions / Basis functions?
 * - Forward evaluation?
 * - Computation of derivatives?
 */

double measure_execution_time(std::function<void()> f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  return elapsed.count();
}

class Timer {
  std::chrono::system_clock::time_point start;

 public:
  Timer() : start(std::chrono::high_resolution_clock::now()){};
  void reset() { start = std::chrono::high_resolution_clock::now(); }
  double elapsed() {
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    return elapsed.count();
  }
};

// using namespace fsi;
int main() {
  constexpr size_t d = 8;
  size_t bound = 24;
  fsi::TemplateBoundedSumIterator<d> it(bound);
  // fsi::BoundedSumIterator it(d, bound);
  std::vector<MonomialFunctions> phi(d);
  std::vector<GoldenPointDistribution> x(d);

  auto rhs = evaluateFunction(it, f, x);
  auto op = createInterpolationOperator(it, phi, x);

  // std::cout << rhs << "\n";

  Timer timer;
  op.prepareSolve();
  // auto result = fsi::interpolate(f, it, phi, x);
  std::cout << "Time for prepareSolve(): " << timer.elapsed() << " s\n";

  timer.reset();
  auto c = op.solve(rhs);
  // auto c = fsi::interpolate(f, it, phi, x);
  std::cout << "Time for solve(): " << timer.elapsed() << " s\n";
  // std::cout << c << "\n";

  timer.reset();
  auto b = op.apply(c);
  std::cout << "Time for apply(): " << timer.elapsed() << " s\n";
  // std::cout << b << "\n";

  std::cout << "Number of points: " << it.numValues() << "\n";

  // for (size_t i = 0; i < result.data.size(); ++i) {
  //   std::cout << result.data[i] << "\n\n";
  // }
}
