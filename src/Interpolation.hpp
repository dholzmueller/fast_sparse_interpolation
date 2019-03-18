// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include <vector>

namespace fsi {

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

  size_t firstIndex() { return index_head[0]; }

  size_t indexAt(size_t dim) { return index_head[dim]; }

  bool done() { return is_done; }

  void reset() {
    index_head = std::vector<size_t>(d - 1, 0);
    index_head_sum = 0;
    is_done = false;
  }

  size_t dim() { return d; }

  std::vector<size_t> indexBounds() { return std::vector<size_t>(d, bound + 1); }
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

// class FastSparseInterpolation {
// public:
//  FastSparseInterpolation(std::function<double(std::vector<double>)> const &f,
//                          BoundedSumIterator it, std::vector<std::function<double(size_t)>> phi);
//};

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

  //  for (size_t i = 0; i < v.data.size(); ++i) {
  //    std::cout << v.data[i] << "\n\n";
  //  }

  std::cout << "First matrix multiplication\n";

  // multiply by L^{-1}
  // the multiplication is based on a cyclic permutation of the indices: the last index of v becomes
  // the first index of w

  for (int k = d - 1; k >= 0; --k) {
    MultiDimVector w(n[k]);
    auto &Lkinv = Linv[k];
    it.reset();

    size_t first_v_index = 0;
    size_t second_v_index = 0;

    while (not it.done()) {
      size_t last_dim_count = it.lastDimensionCount();
      for (size_t i = 0; i < last_dim_count; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j <= i; ++j) {
          // std::cout << Lkinv(i, j) << " * " << v.data[first_v_index][second_v_index + j] << "\n";
          sum += Lkinv(i, j) * v.data[first_v_index][second_v_index + j];
        }
        w.data[i].push_back(sum);
      }
      second_v_index += last_dim_count;
      if (second_v_index >= v.data[first_v_index].size()) {
        second_v_index = 0;
        first_v_index += 1;
      }
      it.next();
    }

    v.swap(w);

    //    for (size_t i = 0; i < v.data.size(); ++i) {
    //      std::cout << v.data[i] << "\n\n";
    //    }
  }

  std::cout << "Second matrix multiplication\n";

  // multiply by U^{-1}
  // the multiplication is based on a cyclic permutation of the indices: the last index of v becomes
  // the first index of w

  for (int k = d - 1; k >= 0; --k) {
    MultiDimVector w(n[k]);
    auto &Ukinv = Uinv[k];
    it.reset();

    size_t first_v_index = 0;
    size_t second_v_index = 0;

    while (not it.done()) {
      size_t last_dim_count = it.lastDimensionCount();
      for (size_t i = 0; i < last_dim_count; ++i) {
        double sum = 0.0;
        for (size_t j = i; j < last_dim_count; ++j) {
          sum += Ukinv(i, j) * v.data[first_v_index][second_v_index + j];
        }
        w.data[i].push_back(sum);
      }
      second_v_index += last_dim_count;
      if (second_v_index >= v.data[first_v_index].size()) {
        second_v_index = 0;
        first_v_index += 1;
      }

      it.next();
    }

    v.swap(w);

    //    for (size_t i = 0; i < v.data.size(); ++i) {
    //      std::cout << v.data[i] << "\n\n";
    //    }
  }

  return v;
}

} /* namespace fsi */
