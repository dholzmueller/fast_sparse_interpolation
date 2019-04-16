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

#pragma once

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include <vector>
#include "Iterators.hpp"

namespace fsi {

/**
 * Represents a vector indexed by multi-indices which are specified by an iterator of type It.
 */
template <typename It>
class MultiDimVector {
  It it;

 public:
  // put the first dimension into an outer vector for processing reasons
  // data[i] contains all entries whose multi-index starts with i, ordered lexicographically.
  std::vector<std::vector<double>> data;

  /**
   * Iterator class used for begin() / end()
   */
  class Iterator {
    MultiDimVector<It> &v;
    StepIterator<It> stepIt;

   public:
    Iterator(MultiDimVector<It> &v, It it) : v(v), stepIt(it){};

    typedef Iterator self_type;
    typedef double value_type;
    typedef double &reference;
    typedef double *pointer;
    typedef std::forward_iterator_tag iterator_category;
    typedef int difference_type;

    std::vector<size_t> index() { return stepIt.index(); }

    // post-increment
    self_type operator++() {
      self_type i = *this;
      stepIt.next();
      return i;
    }

    // pre-increment
    self_type operator++(int junk) {
      stepIt.next();
      return *this;
    }
    reference operator*() { return v.data[stepIt.firstIndex()][stepIt.tailDimsCounter()]; }
    pointer operator->() { return &(*(*this)); }
    bool operator==(const self_type &rhs) {
      if (stepIt.valid() != rhs.stepIt.valid()) {
        return false;
      }

      return stepIt.valid() or (stepIt.firstIndex() == rhs.stepIt.firstIndex() &&
                                stepIt.tailDimsCounter() == rhs.stepIt.tailDimsCounter());
    }
    bool operator!=(const self_type &rhs) { return !(*this == rhs); }
  };

  MultiDimVector(It it) : it(it), data(it.firstIndexBound()) {
    auto sizes = it.numValuesPerFirstIndex();
    for (size_t i = 0; i < sizes.size(); ++i) {
      data[i].resize(sizes[i]);
    }
  };

  void swap(MultiDimVector<It> &other) {
    data.swap(other.data);
    std::swap(it, other.it);
  }

  /**
   * Returns the associated iterator object.
   */
  It getJumpIterator() const { return it; }

  /**
   * Changes the associated iterator object and adjusts the structure of the data vectors (without
   * initializing them).
   */
  void resetWithJumpIterator(It const &other_it) {
    it = other_it;
    size_t n_1 = it.firstIndexBound();
    data.resize(n_1);
    auto sizes = it.numValuesPerFirstIndex();
    for (size_t i = 0; i < n_1; ++i) {
      data[i].resize(sizes[i]);
    }
  }

  Iterator begin() { return Iterator(*this, it); }

  Iterator end() {
    It end_it = it;
    end_it.goToEnd();
    return Iterator(*this, end_it);
  }

  MultiDimVector<It> &operator+=(MultiDimVector<It> const &other) {
    for (size_t i = 0; i < data.size(); ++i) {
      for (size_t j = 0; j < data[i].size(); ++j) {
        data[i][j] += other.data[i][j];
      }
    }
    return *this;
  }

  MultiDimVector<It> &operator-=(MultiDimVector<It> const &other) {
    for (size_t i = 0; i < data.size(); ++i) {
      for (size_t j = 0; j < data[i].size(); ++j) {
        data[i][j] -= other.data[i][j];
      }
    }
    return *this;
  }
};

/**
 * Helper for printing a matrix
 */
inline std::ostream &operator<<(std::ostream &os, boost::numeric::ublas::matrix<double> matrix) {
  os << "[";
  for (size_t i = 0; i < matrix.size1(); ++i) {
    for (size_t j = 0; j < matrix.size2(); ++j) {
      os << matrix(i, j) << "   ";
    }
    os << "\n";
  }
  os << " ]";
  return os;
}

/**
 * Helper for printing a std::vector
 */
template <class T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "[";
  for (auto ii = v.begin(); ii != v.end(); ++ii) {
    os << " " << *ii;
  }
  os << " ]";
  return os;
}

template <typename It>
MultiDimVector<It> operator+(MultiDimVector<It> first, MultiDimVector<It> const &second) {
  first += second;
  return first;
}

template <typename It>
MultiDimVector<It> operator-(MultiDimVector<It> first, MultiDimVector<It> const &second) {
  first -= second;
  return first;
}

template <typename It>
double squared_l2_norm(MultiDimVector<It> const &v) {
  double sum = 0.0;
  for (size_t i = 0; i < v.data.size(); ++i) {
    for (size_t j = 0; j < v.data[i].size(); ++j) {
      double value = v.data[i][j];
      sum += value * value;
    }
  }
  return sum;
}

// ----- The next block of methods contains methods for efficient multi-indexed matrix-vector
// products.

/**
 * Multiplies the multi-indexed vector v with the matrix \widehat{I \otimes \hdots \otimes I \otimes
 * L}, according to the notation from the paper. Saves the result in v. The vector buffer is used
 * for storing intermediate results. It should be provided externally and can be used for multiple
 * matrix-vector products to omit allocating new memory each time.
 * Note that this method permutes indices, specifically it moves the last index to the front. This
 * means e.g. that an element at index (0, 1, 2, 3) of the resulting vector would have been at index
 * (1, 2, 3, 0) if the indices hadn't been permuted.
 * We cycle indices because it allows for almost continuous memory access during each
 * multiplication, hence likely accelerating the method due to better cache performance and better
 * iterator performance.
 */
template <typename It>
void multiply_single_lower_triangular_inplace(boost::numeric::ublas::matrix<double> const &L,
                                              MultiDimVector<It> &v, MultiDimVector<It> &buffer) {
  It it = v.getJumpIterator();
  It cycled_it = it.cycle();
  buffer.resetWithJumpIterator(cycled_it);

  std::vector<size_t> indexes(buffer.data.size(), 0);

  it.reset();

  size_t first_v_index = 0;
  size_t second_v_index = 0;

  double *data_pointer = &v.data[0][0];
  size_t data_size = v.data[0].size();

  while (it.valid()) {
    size_t last_dim_count = it.lastDimensionCount();
    double *offset_data_pointer = data_pointer + second_v_index;
    for (size_t i = 0; i < last_dim_count; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j <= i; ++j) {
        sum += L(i, j) * (*(offset_data_pointer + j));
      }
      buffer.data[i][indexes[i]++] = sum;
    }
    second_v_index += last_dim_count;
    if (second_v_index >= data_size) {
      second_v_index = 0;
      first_v_index += 1;
      data_pointer = &v.data[first_v_index][0];
      data_size = v.data[first_v_index].size();
    }

    it.next();
  }

  v.swap(buffer);
}

/**
 * Same as above, but with an upper triangular matrix U instead of L.
 */
template <typename It>
void multiply_single_upper_triangular_inplace(boost::numeric::ublas::matrix<double> const &U,
                                              MultiDimVector<It> &v, MultiDimVector<It> &buffer) {
  It it = v.getJumpIterator();
  It cycled_it = it.cycle();
  buffer.resetWithJumpIterator(cycled_it);

  std::vector<size_t> indexes(buffer.data.size(), 0);

  it.reset();

  size_t first_v_index = 0;
  size_t second_v_index = 0;

  double *data_pointer = &v.data[0][0];
  size_t data_size = v.data[0].size();

  while (it.valid()) {
    size_t last_dim_count = it.lastDimensionCount();
    double *offset_data_pointer = data_pointer + second_v_index;
    for (size_t i = 0; i < last_dim_count; ++i) {
      double sum = 0.0;
      for (size_t j = i; j < last_dim_count; ++j) {
        sum += U(i, j) * (*(offset_data_pointer + j));
      }
      buffer.data[i][indexes[i]++] = sum;
    }
    second_v_index += last_dim_count;
    if (second_v_index >= data_size) {
      second_v_index = 0;
      first_v_index += 1;
      data_pointer = &v.data[first_v_index][0];
      data_size = v.data[first_v_index].size();
    }

    it.next();
  }

  v.swap(buffer);
}

/**
 * Same as above, but with an arbitrary Matrix M instead of L.
 */
template <typename It>
void multiply_single_arbitrary_inplace(boost::numeric::ublas::matrix<double> const &M,
                                       MultiDimVector<It> &v, MultiDimVector<It> &buffer) {
  It it = v.getJumpIterator();
  It cycled_it = it.cycle();
  buffer.resetWithJumpIterator(cycled_it);

  std::vector<size_t> indexes(buffer.data.size(), 0);

  it.reset();

  size_t first_v_index = 0;
  size_t second_v_index = 0;

  double *data_pointer = &v.data[0][0];
  size_t data_size = v.data[0].size();

  while (it.valid()) {
    size_t last_dim_count = it.lastDimensionCount();
    double *offset_data_pointer = data_pointer + second_v_index;
    for (size_t i = 0; i < last_dim_count; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < last_dim_count; ++j) {
        sum += M(i, j) * (*(offset_data_pointer + j));
      }
      buffer.data[i][indexes[i]++] = sum;
    }
    second_v_index += last_dim_count;
    if (second_v_index >= data_size) {
      second_v_index = 0;
      first_v_index += 1;
      data_pointer = &v.data[first_v_index][0];
      data_size = v.data[first_v_index].size();
    }

    it.next();
  }

  v.swap(buffer);
}

/**
 * Same as above, but multiplies with the identity matrix, i.e. it only cycles indices. This should
 * be somewhat faster than explicitly multiplying with the identity matrix using one of the other
 * three methods.
 */
template <typename It>
void multiply_single_identity_inplace(MultiDimVector<It> &v, MultiDimVector<It> &buffer) {
  It it = v.getJumpIterator();
  It cycled_it = it.cycle();
  buffer.resetWithJumpIterator(cycled_it);

  std::vector<size_t> indexes(buffer.data.size(), 0);

  it.reset();

  size_t first_v_index = 0;
  size_t second_v_index = 0;

  double *data_pointer = &v.data[0][0];
  size_t data_size = v.data[0].size();

  while (it.valid()) {
    size_t last_dim_count = it.lastDimensionCount();
    double *offset_data_pointer = data_pointer + second_v_index;
    for (size_t i = 0; i < last_dim_count; ++i) {
      buffer.data[i][indexes[i]++] = (*(offset_data_pointer + i));
    }
    second_v_index += last_dim_count;
    if (second_v_index >= data_size) {
      second_v_index = 0;
      first_v_index += 1;
      data_pointer = &v.data[first_v_index][0];
      data_size = v.data[first_v_index].size();
    }

    it.next();
  }

  v.swap(buffer);
}

/**
 * Repeats calls to the identity multiplication to cycle the indices several times.
 */
template <typename It>
void cycle_vector_inplace(size_t num_times, MultiDimVector<It> &v, MultiDimVector<It> &buffer) {
  for (size_t i = 0; i < num_times; ++i) {
    multiply_single_identity_inplace(v, buffer);
  }
}

/**
 * Repeats calls to the identity multiplication to cycle the indices several times. Uses its own
 * buffer (potentially slower than reusing an already created buffer).
 */
template <typename It>
void cycle_vector_inplace(size_t num_times, MultiDimVector<It> &v) {
  MultiDimVector<It> buffer(v.getJumpIterator());
  for (size_t i = 0; i < num_times; ++i) {
    multiply_single_identity_inplace(v, buffer);
  }
}

/**
 * Performs a multiplication with a tensor product of lower triangular matrices \widehat{L[1]
 * \otimes \hdots \otimes L[k]}.
 */
template <typename It>
void multiply_lower_triangular_inplace(std::vector<boost::numeric::ublas::matrix<double>> L,
                                       MultiDimVector<It> &v, MultiDimVector<It> &buffer) {
  // the multiplication is based on a cyclic permutation of the indices: the last index of v becomes
  // the first index of w
  size_t d = L.size();
  It it = v.getJumpIterator();

  for (int k = d - 1; k >= 0; --k) {
    multiply_single_lower_triangular_inplace(L[k], v, buffer);
  }
}

/**
 * Performs a multiplication with a tensor product of upper triangular matrices \widehat{U[1]
 * \otimes \hdots \otimes U[k]}.
 */
template <typename It>
void multiply_upper_triangular_inplace(std::vector<boost::numeric::ublas::matrix<double>> U,
                                       MultiDimVector<It> &v, MultiDimVector<It> &buffer) {
  // the multiplication is based on a cyclic permutation of the indices: the last index of v becomes
  // the first index of w
  size_t d = U.size();
  It it = v.getJumpIterator();

  for (int k = d - 1; k >= 0; --k) {
    multiply_single_upper_triangular_inplace(U[k], v, buffer);
  }
}

/**
 * Represents a sparse linear tensor product operator defined by a matrix for each dimension and an
 * iterator that defines the multi-index set.
 */
template <typename It>
class SparseTPOperator {
  // Iterator that defines the multi-index set
  It it;

  // Matrices associated with the operator.
  std::vector<boost::numeric::ublas::matrix<double>> M;

  // Matrices containing the L and U parts of the LU decompositions
  std::vector<boost::numeric::ublas::matrix<double>> LU;

  // Matrices containing the L parts of the LU decompositions
  std::vector<boost::numeric::ublas::matrix<double>> L;

  // Matrices containing the U parts of the LU decomposition
  std::vector<boost::numeric::ublas::matrix<double>> U;

  // Inverses of the L matrices
  std::vector<boost::numeric::ublas::matrix<double>> Linv;

  // Inverses of the U matrices
  std::vector<boost::numeric::ublas::matrix<double>> Uinv;

  // is used to store intermediate data
  MultiDimVector<It> buffer;

  // dimension of the indices
  size_t d;

 public:
  SparseTPOperator(It it, std::vector<boost::numeric::ublas::matrix<double>> matrices)
      : it(it), M(matrices), buffer(it), d(matrices.size()){};

  /**
   * This method performs preparations common to apply() and solve() if they haven't already been
   * performed. The preparations are the LU decompositions of the M matrices.
   */
  void prepareCommon() {
    if (LU.size() > 0) {
      return;  // already prepared
    }

    namespace ublas = boost::numeric::ublas;
    typedef ublas::matrix<double> Matrix;

    for (size_t k = 0; k < d; ++k) {
      // std::cout << "Matrix creation loop\n";
      Matrix Mk = M[k];

      ublas::lu_factorize(Mk);

      LU.push_back(Mk);
    }
  }

  /**
   * This method performs preparations specific to apply() if they haven't already been performed.
   * These are the (cheap) computation of L and U from LU.
   */
  void prepareApply() {
    if (L.size() > 0) {
      return;  // already prepared
    }

    prepareCommon();

    namespace ublas = boost::numeric::ublas;
    typedef ublas::matrix<double> Matrix;

    for (size_t k = 0; k < d; ++k) {
      size_t nk = M[k].size1();
      Matrix &LUk = LU[k];
      Matrix Lk(nk, nk);
      Matrix Uk(nk, nk);

      for (size_t i = 0; i < nk; ++i) {
        for (size_t j = 0; j < i; ++j) {
          Lk(i, j) = LUk(i, j);
        }

        Lk(i, i) = 1.0;

        for (size_t j = i; j < nk; ++j) {
          Uk(i, j) = LUk(i, j);
        }
      }

      L.push_back(Lk);
      U.push_back(Uk);
    }
  }

  /**
   * This method performs preparations specific to solve() if they haven't already been performed.
   * This is the inversion of the L and U matrices.
   */
  void prepareSolve() {
    if (Linv.size() > 0) {
      return;  // already prepared
    }

    prepareCommon();

    namespace ublas = boost::numeric::ublas;
    typedef ublas::matrix<double> Matrix;

    for (size_t k = 0; k < d; ++k) {
      Matrix Lkinv = ublas::identity_matrix<double>(M[k].size1());
      Matrix Ukinv = ublas::identity_matrix<double>(M[k].size1());

      ublas::inplace_solve(LU[k], Lkinv, ublas::unit_lower_tag());
      ublas::inplace_solve(LU[k], Ukinv, ublas::upper_tag());

      Linv.push_back(Lkinv);
      Uinv.push_back(Ukinv);
    }
  }

  /**
   * Performs a matrix-vector product with the matrix that is implicitly defined by this operator.
   */
  MultiDimVector<It> apply(MultiDimVector<It> input) {
    prepareApply();
    multiply_upper_triangular_inplace(U, input, buffer);
    multiply_lower_triangular_inplace(L, input, buffer);
    return input;
  }

  /**
   * Performs a matrix-vector product with the inverse of the matrix that is implicitly defined by
   * this operator.
   */
  MultiDimVector<It> solve(MultiDimVector<It> rhs) {
    prepareSolve();
    multiply_lower_triangular_inplace(Linv, rhs, buffer);
    multiply_upper_triangular_inplace(Uinv, rhs, buffer);
    return rhs;
  }
};

/**
 * Creates a SparseTPOperator corresponding to the interpolation problem. That means its
 * one-dimensional matrices contain the entries phi[k](j)(x[k](i)). The type of phi and x must be
 * such that this expression is defined.
 */
template <typename It, typename X, typename Phi>
SparseTPOperator<It> createInterpolationOperator(It it, Phi phi, X x) {
  auto n = it.indexBounds();
  size_t d = it.dim();

  namespace ublas = boost::numeric::ublas;
  typedef ublas::matrix<double> Matrix;

  std::vector<Matrix> matrices;

  // create matrices and inverted LU decompositions
  for (size_t k = 0; k < d; ++k) {
    // std::cout << "Matrix creation loop\n";
    Matrix Mk(n[k], n[k]);
    for (size_t i = 0; i < n[k]; ++i) {
      for (size_t j = 0; j < n[k]; ++j) {
        Mk(i, j) = phi[k](j)(x[k](i));
      }
    }

    matrices.push_back(Mk);
  }

  return SparseTPOperator<It>(it, matrices);
}

/**
 * Creates a MultiDimVector containing the values of the function f at the grid points specified by
 * x and the multi-index iterator it. The function f should take a std::vector<double> as an
 * argument. The point object x should have a type such that the expression x[k](i) yields a double
 * specifying the i-th point (starting from i=0) in dimension k (also starting from k=0).
 */
template <typename It, typename Func, typename X>
MultiDimVector<It> evaluateFunction(It it, Func f, X x) {
  size_t d = it.dim();
  auto n = it.indexBounds();
  MultiDimVector<It> v(it);
  std::vector<size_t> indexes(v.data.size(), 0);

  it.reset();
  std::vector<double> point(d);
  while (it.valid()) {
    size_t last_dim_count = it.lastDimensionCount();
    for (size_t dim = 0; dim < d - 1; ++dim) {
      point[dim] = x[dim](it.indexAt(dim));
    }

    for (size_t last_dim_idx = 0; last_dim_idx < last_dim_count; ++last_dim_idx) {
      point[d - 1] = x[d - 1](last_dim_idx);

      double function_value = f(point);

      size_t first_index = it.firstIndex();
      v.data[first_index][indexes[first_index]++] = function_value;
    }

    it.next();
  }

  return v;
}

/**
 * Convenience function that computes a vector of interpolation coefficients for the sparse grid
 * basis given by phi and it on the points given by phi and it. The arguments f, it, phi, x need to
 * satisfy the same requirements as explained in the documentation of the functions above.
 */
template <typename Func, typename It, typename Phi, typename X>
MultiDimVector<It> interpolate(Func f, It it, Phi phi, X x) {
  auto rhs = evaluateFunction(it, f, x);
  auto op = createInterpolationOperator(it, phi, x);
  return op.solve(rhs);
}

} /* namespace fsi */
