// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <Interpolation.hpp>
#include <Iterators.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

inline double f(std::vector<double> point) {
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

inline std::ostream &operator<<(std::ostream &os, fsi::MultiDimVector const &v) {
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

inline double measure_execution_time(std::function<void()> f) {
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

void measurePerformance();
