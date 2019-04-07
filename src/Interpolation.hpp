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

template <typename Func, typename It, typename Phi, typename X>
MultiDimVector interpolate(Func f, It it, Phi phi, X x) {
  auto rhs = evaluateFunction(it, f, x);
  auto op = createInterpolationOperator(it, phi, x);
  return op.solve(rhs);
}

} /* namespace fsi */
