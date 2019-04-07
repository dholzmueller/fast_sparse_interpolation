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

namespace fsi {

size_t binom(size_t n, size_t k) {
  if (2 * k > n) {
    k = n - k;
  }

  size_t prod = 1;
  size_t upper_factor = n + 1 - k;
  size_t lower_factor = 1;
  while (upper_factor <= n) {
    prod *= upper_factor;
    prod /= lower_factor;
    upper_factor += 1;
    lower_factor += 1;
  }
  return prod;
}

template <size_t d>
class TemplateBoundedSumIterator {
  size_t bound;
  std::vector<size_t> index_head;  // contains all entries of the index except the last one
  size_t index_head_sum;
  bool is_done;

 public:
  TemplateBoundedSumIterator(size_t bound)
      : bound(bound), index_head(d - 1, 0), index_head_sum(0), is_done(false){};

  /**
   * At the current multi-index (i_1, ..., i_{d-1}, 0), return how many multi-indices starting with
   * (i_1, ..., i_{d-1}) are contained in the multi-index set, then advance to the next multi-index
   * that ends with a zero.
   */
  size_t lastDimensionCount() { return bound - index_head_sum + 1; }

  void next() {
    if (bound > index_head_sum) {
      index_head_sum += 1;
      index_head[d - 2] += 1;
    } else {
      int dim = d - 2;

      for (; dim >= 0 && index_head[dim] == 0; --dim) {
        // reduce dimension until entry is nonzero
      }

      if (dim > 0) {
        index_head_sum -= (index_head[dim] - 1);
        index_head[dim] = 0;
        index_head[dim - 1] += 1;
      } else if (dim == 0) {
        index_head[dim] = 0;
        index_head_sum = 0;
        is_done = true;
      }
    }
  }

  size_t firstIndex() const { return index_head[0]; }

  size_t indexAt(size_t dim) const { return index_head[dim]; }

  bool done() const { return is_done; }

  void reset() {
    index_head = std::vector<size_t>(d - 1, 0);
    index_head_sum = 0;
    is_done = false;
  }

  size_t dim() const { return d; }

  std::vector<size_t> indexBounds() const { return std::vector<size_t>(d, bound + 1); }

  size_t numValues() const { return binom(bound + d, d); }

  /**
   * Returns an iterator where the last index moves to the front. For an index set defined by a sum
   * bound, nothing changes.
   */
  TemplateBoundedSumIterator<d> cycle() const {
    TemplateBoundedSumIterator<d> it = *this;
    it.reset();
    return it;
  }
};

class BoundedSumIterator {
  size_t d;
  size_t bound;
  std::vector<size_t> index_head;  // contains all entries of the index except the last one
  size_t index_head_sum;
  bool is_done;

 public:
  BoundedSumIterator(size_t d, size_t bound)
      : d(d), bound(bound), index_head(d - 1, 0), index_head_sum(0), is_done(false){};

  /**
   * At the current multi-index (i_1, ..., i_{d-1}, 0), return how many multi-indices starting with
   * (i_1, ..., i_{d-1}) are contained in the multi-index set, then advance to the next multi-index
   * that ends with a zero.
   */
  size_t lastDimensionCount() { return bound - index_head_sum + 1; }

  void next() {
    if (bound > index_head_sum) {
      index_head_sum += 1;
      index_head[d - 2] += 1;
    } else {
      int dim = d - 2;

      for (; dim >= 0 && index_head[dim] == 0; --dim) {
        // reduce dimension until entry is nonzero
      }

      if (dim > 0) {
        index_head_sum -= (index_head[dim] - 1);
        index_head[dim] = 0;
        index_head[dim - 1] += 1;
      } else if (dim == 0) {
        index_head[dim] = 0;
        index_head_sum = 0;
        is_done = true;
      }
    }
  }

  size_t firstIndex() const { return index_head[0]; }

  size_t indexAt(size_t dim) const { return index_head[dim]; }

  bool done() const { return is_done; }

  void reset() {
    index_head = std::vector<size_t>(d - 1, 0);
    index_head_sum = 0;
    is_done = false;
  }

  size_t dim() const { return d; }

  std::vector<size_t> indexBounds() const { return std::vector<size_t>(d, bound + 1); }

  size_t numValues() const { return binom(bound + d, d); }

  /**
   * Returns an iterator where the last index moves to the front. For an index set defined by a sum
   * bound, nothing changes.
   */
  BoundedSumIterator cycle() {
    BoundedSumIterator it = *this;
    it.reset();
    return it;
  }
};

class StandardBoundedSumIterator {
  size_t d;
  size_t bound;
  std::vector<size_t> index;  // contains all entries of the index except the last one
  size_t index_sum;
  bool is_done;

 public:
  StandardBoundedSumIterator(size_t d, size_t bound)
      : d(d), bound(bound), index(d, 0), index_sum(0), is_done(false){};

  /**
   * At the current multi-index (i_1, ..., i_{d-1}, 0), return how many multi-indices starting with
   * (i_1, ..., i_{d-1}) are contained in the multi-index set, then advance to the next multi-index
   * that ends with a zero.
   */
  bool next() {
    if (bound > index_sum) {
      index_sum += 1;
      index[d - 1] += 1;
    } else {
      int dim = d - 1;

      for (; dim >= 0 && index[dim] == 0; --dim) {
        // reduce dimension until entry is nonzero
      }

      if (dim > 0) {
        index_sum -= index[dim];
        index[dim] = 0;
        index[dim - 1] += 1;
      } else if (dim == 0) {
        index[dim] = 0;
        index_sum = 0;
        is_done = true;
      }
    }
    return is_done;
  }

  size_t firstIndex() { return index[0]; }

  size_t indexSum() { return index_sum; }

  bool done() { return is_done; }

  void reset() {
    index = std::vector<size_t>(d - 1, 0);
    index_sum = 0;
    is_done = false;
  }
};

class MultiDimVector {
 public:
  // put the first dimension into an outer vector for processing reasons
  std::vector<std::vector<double>> data;
  MultiDimVector(size_t n) : data(n){};

  void swap(MultiDimVector &other) { data.swap(other.data); }

  void clear() {
    for (size_t dim = 0; dim < data.size(); ++dim) {
      data[dim].clear();
    }
  }
};

template <typename It>
void multiply_lower_triangular_inplace(It it, std::vector<boost::numeric::ublas::matrix<double>> L,
                                       MultiDimVector &v) {
  // the multiplication is based on a cyclic permutation of the indices: the last index of v becomes
  // the first index of w
  size_t d = L.size();
  auto n = it.indexBounds();

  for (int k = d - 1; k >= 0; --k) {
    MultiDimVector w(n[k]);
    for (size_t idx = 0; idx < n[k]; ++idx) {
      // TODO: make compatible with other iterators
      w.data[idx].reserve(v.data[idx].size());
    }

    auto &Lk = L[k];
    it.reset();

    size_t first_v_index = 0;
    size_t second_v_index = 0;

    double *data_pointer = &v.data[0][0];
    size_t data_size = v.data[0].size();

    while (not it.done()) {
      size_t last_dim_count = it.lastDimensionCount();
      double *offset_data_pointer = data_pointer + second_v_index;
      for (size_t i = 0; i < last_dim_count; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j <= i; ++j) {
          sum += Lk(i, j) * (*(offset_data_pointer + j));
        }
        w.data[i].push_back(sum);
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

    v.swap(w);

    it = it.cycle();

    //    for (size_t i = 0; i < v.data.size(); ++i) {
    //      std::cout << v.data[i] << "\n\n";
    //    }
  }
}

template <typename It>
void multiply_upper_triangular_inplace(It it, std::vector<boost::numeric::ublas::matrix<double>> U,
                                       MultiDimVector &v) {
  // the multiplication is based on a cyclic permutation of the indices: the last index of v becomes
  // the first index of w
  size_t d = U.size();
  auto n = it.indexBounds();

  for (int k = d - 1; k >= 0; --k) {
    MultiDimVector w(n[k]);
    for (size_t idx = 0; idx < n[k]; ++idx) {
      // TODO: make compatible with other iterators
      w.data[idx].reserve(v.data[idx].size());
    }

    auto &Uk = U[k];
    it.reset();

    size_t first_v_index = 0;
    size_t second_v_index = 0;

    double *data_pointer = &v.data[0][0];
    size_t data_size = v.data[0].size();

    while (not it.done()) {
      size_t last_dim_count = it.lastDimensionCount();
      double *offset_data_pointer = data_pointer + second_v_index;
      for (size_t i = 0; i < last_dim_count; ++i) {
        double sum = 0.0;
        for (size_t j = i; j < last_dim_count; ++j) {
          sum += Uk(i, j) * (*(offset_data_pointer + j));
        }
        w.data[i].push_back(sum);
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

    it = it.cycle();

    v.swap(w);

    //    for (size_t i = 0; i < v.data.size(); ++i) {
    //      std::cout << v.data[i] << "\n\n";
    //    }
  }
}

/**
 * Represents a sparse linear tensor product operator defined by a matrix for each dimension.
 */
template <typename It>
class SparseTPOperator {
  It it;
  std::vector<boost::numeric::ublas::matrix<double>> M;
  std::vector<boost::numeric::ublas::matrix<double>> LU;
  std::vector<boost::numeric::ublas::matrix<double>> L;
  std::vector<boost::numeric::ublas::matrix<double>> U;
  std::vector<boost::numeric::ublas::matrix<double>> Linv;
  std::vector<boost::numeric::ublas::matrix<double>> Uinv;
  size_t d;

 public:
  SparseTPOperator(It it, std::vector<boost::numeric::ublas::matrix<double>> matrices)
      : it(it), M(matrices), d(matrices.size()){};

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

  MultiDimVector apply(MultiDimVector input) {
    prepareApply();
    multiply_upper_triangular_inplace(it, U, input);
    multiply_lower_triangular_inplace(it, L, input);
    return input;
  }

  MultiDimVector solve(MultiDimVector rhs) {
    prepareSolve();
    multiply_lower_triangular_inplace(it, Linv, rhs);
    multiply_upper_triangular_inplace(it, Uinv, rhs);
    return rhs;
  }
};

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

template <typename It, typename Func, typename X>
MultiDimVector evaluateFunction(It it, Func f, X x) {
  size_t d = it.dim();
  auto n = it.indexBounds();
  MultiDimVector v(n[0]);

  it.reset();
  std::vector<double> point(d);
  while (not it.done()) {
    size_t last_dim_count = it.lastDimensionCount();
    for (size_t dim = 0; dim < d - 1; ++dim) {
      point[dim] = x[dim](it.indexAt(dim));
    }

    for (size_t last_dim_idx = 0; last_dim_idx < last_dim_count; ++last_dim_idx) {
      point[d - 1] = x[d - 1](last_dim_idx);

      double function_value = f(point);

      v.data[it.firstIndex()].push_back(function_value);
    }

    it.next();
  }

  return v;
}

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

template <class T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "[";
  for (auto ii = v.begin(); ii != v.end(); ++ii) {
    os << " " << *ii;
  }
  os << " ]";
  return os;
}

template <typename Func, typename It, typename Phi, typename X>
MultiDimVector interpolate(Func f, It it, Phi phi, X x) {
  auto n = it.indexBounds();
  size_t d = it.dim();

  namespace ublas = boost::numeric::ublas;
  typedef ublas::matrix<double> Matrix;

  std::vector<Matrix> Linv, Uinv;

  // create matrices and inverted LU decompositions
  for (size_t k = 0; k < d; ++k) {
    // std::cout << "Matrix creation loop\n";
    Matrix Mk(n[k], n[k]);
    for (size_t i = 0; i < n[k]; ++i) {
      for (size_t j = 0; j < n[k]; ++j) {
        Mk(i, j) = phi[k](j)(x[k](i));
      }
    }

    // std::cout << "Matrices:\n";

    // std::cout << Mk << "\n";

    ublas::lu_factorize(Mk);

    // std::cout << Mk << "\n";

    Matrix Lkinv = ublas::identity_matrix<double>(n[k]);
    Matrix Ukinv = ublas::identity_matrix<double>(n[k]);
    ublas::inplace_solve(Mk, Lkinv, ublas::unit_lower_tag());
    ublas::inplace_solve(Mk, Ukinv, ublas::upper_tag());

    // std::cout << Lkinv << "\n";

    // std::cout << Ukinv << "\n";

    Linv.push_back(Lkinv);
    Uinv.push_back(Ukinv);
  }

  MultiDimVector v(n[0]);

  std::cout << "Compute function values\n";

  // compute function values
  it.reset();
  std::vector<double> point(d);
  while (not it.done()) {
    size_t last_dim_count = it.lastDimensionCount();
    for (size_t dim = 0; dim < d - 1; ++dim) {
      point[dim] = x[dim](it.indexAt(dim));
    }

    for (size_t last_dim_idx = 0; last_dim_idx < last_dim_count; ++last_dim_idx) {
      point[d - 1] = x[d - 1](last_dim_idx);

      double function_value = f(point);

      v.data[it.firstIndex()].push_back(function_value);
    }

    it.next();
  }

  size_t number = 0;
  for (size_t dim = 0; dim < n[0]; ++dim) {
    number += v.data[dim].size();
  }
  std::cout << "number of points: " << number << "\n";
  std::cout << "alternative number of points: " << it.numValues() << "\n";

  //  for (size_t i = 0; i < v.data.size(); ++i) {
  //    std::cout << v.data[i] << "\n\n";
  //  }

  std::cout << "First matrix multiplication\n";

  // multiply by L^{-1}

  multiply_lower_triangular_inplace(it, Linv, v);

  std::cout << "Second matrix multiplication\n";

  // multiply by U^{-1}
  // the multiplication is based on a cyclic permutation of the indices: the last index of v becomes
  // the first index of w

  multiply_upper_triangular_inplace(it, Uinv, v);

  return v;
}

} /* namespace fsi */
