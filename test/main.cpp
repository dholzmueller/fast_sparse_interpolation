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

/**
 * Possible additions:
 * - Tests
 * - More point distributions / Basis functions
 * - Computation of derivatives
 */

/**
 * Simple interpolation code. The function to evaluate is f(x, y, z) = x*y*z, so it can be
 * perfectly interpolated using the tensor product of the linear monomials at index (1, 1, 1).
 */
void example1() {
  // Choose a dimensionality of the problem.
  size_t d = 3;

  // As multi-indices indexing our points and basis functions, this implementation provides an
  // iterator for the standard index set
  // {(i_1, ..., i_d) | i_1, ..., i_d >= 0, i_1 + ... + i_d <= bound}.
  // We configure the bound here.
  size_t bound = 3;

  // This is the iterator that (implicitly) specifies the multi-index set used for the grid points
  // and basis functions.
  fsi::BoundedSumIterator it(d, bound);

  // This is a vector of function-like objects such that we can write phi[k](i)(x) for the i-th
  // basis function (size_t i >= 0) in dimension k, evaluated at the point x (double).
  // As an example, we provide Monomials (phi[k](i)(x) = x^i).
  // Note that the Vandermonde matrix (the interpolation matrix) for monomials is badly conditioned.
  // Therefore, the result gets quite bad when we use bounds >= 30.
  // For this purpose, it would be better to use Legendre polynomials or similar things.
  std::vector<MonomialFunctions> phi(d);

  // This is a vector of function-like objects such that we can write x[k](i) for the i-th point
  // (size_t i >= 0) in dimension k. This distribution is only an example, in practice things like
  // uniform distributions for
  std::vector<GoldenPointDistribution> x(d);

  // Determine a (multi-dimensional) vector with the interpolation coefficients.
  auto coefficients = fsi::interpolate(f, it, phi, x);

  // Print the coefficients and their multi-indices. These multi-indices are those that the iterator
  // it implicitly specifies.
  for (auto coeff_it = coefficients.begin(); coeff_it != coefficients.end(); ++coeff_it) {
    std::cout << "Coefficient at index " << coeff_it.index() << ": " << *coeff_it << "\n";
  }
}

/**
 * More detailed interpolation and evaluation code.
 */
void example2() {
  size_t d = 8;
  size_t bound = 24;
  fsi::BoundedSumIterator it(d, bound);
  std::vector<MonomialFunctions> phi(d);
  std::vector<GoldenPointDistribution> x(d);

  // Create a vector containing the function values at the grid points
  auto rhs = fsi::evaluateFunction(it, f, x);

  // Create an object representing the linear sparse tensor product matrix corresponding to the
  // interpolation problem
  auto op = fsi::createInterpolationOperator(it, phi, x);

  // Run the LU decomposition and invert L and U. If this function is not called, it is
  // automatically called once you call solve(). It can be called separately for performance
  // measurements. This preparation will only done once for an operator object, no matter how often
  // solve() or prepareSolve() is called.
  op.prepareSolve();

  // Solve the linear system with the vector of function values as a right hand side to get a vector
  // of coefficients.
  auto coefficients = op.solve(rhs);

  // We can also prepare the apply operation, which will do almost nothing here since the LU
  // decomposition has already been done. As op.prepareSolve(), this is not necessary.
  op.prepareApply();

  // The apply operation performs the matrix-vector product of the operator's sparse tensor product
  // matrix with the coefficients vector. In our case, this means that we compute the values of the
  // interpolant at the grid points based on its coefficients.
  auto b = op.apply(coefficients);

  // Verify that solve and apply are numerically stable, i.e. that b and rhs are (almost) equal.
  std::cout << "Reconstruction error (L2 norm): " << sqrt(fsi::squared_l2_norm(b - rhs)) << "\n";

  // Print the number of grid points (which equals the number of basis functions) that are used in
  // total.
  std::cout << "Number of points: " << it.numValues() << "\n";
}

/**
 * Similar to example 2, but includes timing information.
 */
void runFunctions() {
  size_t d = 8;
  size_t bound = 30;
  fsi::BoundedSumIterator it(d, bound);
  std::vector<MonomialFunctions> phi(d);
  std::vector<GoldenPointDistribution> x(d);

  auto rhs = evaluateFunction(it, f, x);
  auto op = createInterpolationOperator(it, phi, x);

  // std::cout << rhs << "\n";

  Timer timer;
  timer.reset();
  op.prepareSolve();
  std::cout << "Time for prepareSolve(): " << timer.elapsed() << " s\n";

  timer.reset();
  auto c = op.solve(rhs);
  std::cout << "Time for solve(): " << timer.elapsed() << " s\n";
  // std::cout << c << "\n";

  timer.reset();
  auto b = op.apply(c);
  std::cout << "Time for apply(): " << timer.elapsed() << " s\n";
  // std::cout << b << "\n";

  // timer.reset();
  // fsi::cycle_vector_inplace(d, b);
  // std::cout << "Time for cycling: " << timer.elapsed() << "s\n";

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
  // runFunctions();
  // measurePerformance();
  // example1();
  example2();
}
