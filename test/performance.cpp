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

  std::vector<size_t> dimensions = {2, 4, 8, 16, 32, 64};
  // size_t maxNumPoints = 20000000;
  size_t maxNumPoints = 30000000;

  std::ostringstream stream;

  // minimum quotient of number of points of current configuration to
  // previous configuration
  double minQuotient = 4.0;

  // maximum runtime in seconds
  // if exceeded, we do not try higher numbers of points
  double maxRuntime = 2.0;

  // small value for denominator of a fraction
  double epsilon = 0.01;

  // perform 2*k+1 measurements per configuration, then take median
  size_t k = 2;

  // warm-up

  std::cout << "Warm-up\n";

  size_t d_warmup = 5;
  size_t bound_warmup = 60;

  fsi::BoundedSumIterator it(d_warmup, bound_warmup);
  std::vector<MonomialFunctions> phi(d_warmup);
  std::vector<GoldenPointDistribution> x(d_warmup);

  auto rhs = evaluateFunction(it, f, x);
  auto op = createInterpolationOperator(it, phi, x);

  op.solve(rhs);

  std::cout << "Warm-up finished\n";

  for (size_t d : dimensions) {
    std::cout << "Dimension: " << d << "\n";

    size_t last_num_points = 0;
    double last_runtime = 0.0;

    for (size_t bound = 1; true; ++bound) {
      size_t num_points = fsi::binom(bound + d, d);
      if (num_points > maxNumPoints) break;
      if (num_points / (last_num_points + epsilon) * last_runtime > maxRuntime) break;
      if (num_points < last_num_points * minQuotient) continue;

      last_num_points = num_points;

      std::cout << "Bound: " << bound << "\n";
      double runtime = measureRuntime(
          [&]() {
            fsi::BoundedSumIterator it(d, bound);
            std::vector<MonomialFunctions> phi(d);
            std::vector<GoldenPointDistribution> x(d);

            auto rhs = evaluateFunction(it, f, x);
            auto op = createInterpolationOperator(it, phi, x);

            Timer timer;
            timer.reset();
            op.solve(rhs);
            double time = timer.elapsed();
            std::cout << "Time for solve(): " << time << " s\n";
            return time;
          },
          k);

      stream << d << ", " << bound << ", " << fsi::binom(bound + d, d) << ", " << runtime << "\n";

      last_runtime = runtime;
    }
  }

  std::ofstream file("performance_data.csv");
  file << stream.str();
}
