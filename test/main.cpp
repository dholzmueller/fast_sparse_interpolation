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

// TODO: refactoring:
/**
 * - Tests?
 * - More point distributions / Basis functions?
 * - Computation of derivatives?
 * - Examples?
 */

void runFunctions() {
  constexpr size_t d = 30;
  size_t bound = 8;
  // fsi::TemplateBoundedSumIterator<d> it(bound);
  fsi::BoundedSumIterator it(d, bound);
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

  std::cout << "Reconstruction error (L2 norm): " << sqrt(fsi::squared_l2_norm(b - rhs)) << "\n";

  //  for (auto it = c.begin(); it != c.end(); ++it) {
  //    std::cout << "Value at index " << it.index() << ": " << *it << "\n";
  //  }

  // for (size_t i = 0; i < result.data.size(); ++i) {
  //   std::cout << result.data[i] << "\n\n";
  // }
}

// using namespace fsi;
int main() {
  runFunctions();
  // measurePerformance();
}
