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

#include "common.hpp"

#include <fstream>

double measureRuntime(std::function<double()> f, size_t k = 3) {
  size_t n = 2 * k + 1;

  std::vector<double> singleResults(n);
  for (size_t i = 0; i < n; ++i) {
    singleResults[i] = f();
  }
  std::sort(singleResults.begin(), singleResults.end());
  // return (singleResults[k-1] + singleResults[k] +
  // singleResults[k+1]) / 3.0;
  return singleResults[k];
}

void measurePerformance() {
  // std::vector < size_t > dimensions = {2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 };
  // std::vector<size_t> dimensions = {5, 10};

  std::vector<size_t> dimensions = {2, 4, 8, 16, 32};
  // size_t maxNumPoints = 20000000;
  size_t maxNumPoints = 500000;

  std::ostringstream stream;

  size_t approxNumSteps = 8;
  size_t k = 1;

  for (size_t d : dimensions) {
    std::cout << "Dimension: " << d << "\n";
    size_t maxBound = 0;

    while (fsi::binom(maxBound + d, d) <= maxNumPoints) {
      maxBound += 1;
    }

    size_t stepsize = maxBound / approxNumSteps + 1;

    for (size_t bound = 2; bound < maxBound; bound += stepsize) {
      std::cout << "Bound: " << bound << "\n";
      double runtime = measureRuntime(
          [&]() {
            fsi::BoundedSumIterator it(d, bound);
            std::vector<MonomialFunctions> phi(d);
            std::vector<GoldenPointDistribution> x(d);

            auto rhs = evaluateFunction(it, f, x);
            auto op = createInterpolationOperator(it, phi, x);

            Timer timer;
            op.solve(rhs);
            double time = timer.elapsed();
            std::cout << "Time for solve(): " << time << " s\n";
            return time;
          },
          k);

      stream << d << ", " << bound << ", " << fsi::binom(bound + d, d) << ", " << runtime << "\n";
    }
  }

  std::ofstream file("performance_data.csv");
  file << stream.str();
}
