// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <vector>

namespace fsi {

/**
 * Computes a binomial coefficient.
 */
inline size_t binom(size_t n, size_t k) {
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

/**
 * An iterator class for iterating lexicographically over a multi-index set of the form
 * {(i_1, ..., i_d) | i_1, ..., i_d >= 0, i_1 + ... + i_d <= bound}. Other iterators for other index
 * sets may be implemented following the interface of this iterator class.
 *
 * Since the fast matrix-vector product sums over the last dimension, this iterator does not iterate
 * over the last dimension. For example, if d = 3 and bound = 2, the index set is (lexicographically
 * ordered)
 * {(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 2, 0), (1, 0, 0), (1, 0, 1), (1, 1,
 * 0), (2, 0, 0)}. The iterator iterates over the indices
 * {(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0)} and for each index provides a
 * number n >= 1 that indicates how many values the last dimension can take. For example, when the
 * iterator is at the index (0, 1, 0), we would have n = 2 since (0, 1, 0) and (0, 1, 1) belong to
 * the index set. In contrast, we would have n = 3 for (0, 0, 0) and n = 1 for (2, 0, 0).
 */
class BoundedSumIterator {
  size_t d;
  size_t bound;

  // contains all entries of the index except the last one
  std::vector<size_t> index_head;

  // contains the sum of the entries of index_head
  size_t index_head_sum;

  // stores whether the iterator is still at a valid position.
  // This is set to false whenever the iterator is in the state (bound, 0, ..., 0) and next() is
  // called.
  bool is_valid;

 public:
  /**
   * Constructor with dimension d of the indices (d includes the dimension that is not iterated
   * over) and the bound for the sum of indices. Initially, the iterator points to the multi-index
   * (0, ..., 0).
   */
  BoundedSumIterator(size_t d, size_t bound)
      : d(d), bound(bound), index_head(d - 1, 0), index_head_sum(0), is_valid(true){};

  /**
   * At the current multi-index (i_1, ..., i_{d-1}, 0), return how many multi-indices starting with
   * (i_1, ..., i_{d-1}) are contained in the multi-index set, then advance to the next multi-index
   * that ends with a zero. This corresponds to the value n in the class description.
   */
  size_t lastDimensionCount() { return bound - index_head_sum + 1; }

  /**
   * Advances to the next multi-index ending with zero.
   */
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
        index_head[0] = 0;
        index_head_sum = 0;
        is_valid = false;
      }
    }
  }

  /**
   * Returns the component i_1 of the current multi-index (i_1, ..., i_{d-1}, 0).
   */
  size_t firstIndex() const { return index_head[0]; }

  /**
   * Returns the component of the multi-index at dimension dim, where dim < d-1.
   */
  size_t indexAt(size_t dim) const { return index_head[dim]; }

  /**
   * Returns the vector (i_1, ..., i_{d-1}) where (i_1, \hdots, i_{d-1}, 0) is the current
   * multi-index.
   */
  std::vector<size_t> getIndexHead() const { return index_head; }

  /**
   * Returns true if the iterator points to a valid multi-index (i.e. the iteration is not
   * finished).
   */
  bool valid() const { return is_valid; }

  /**
   * Resets the iterator to (0, ..., 0) and resets valid() to true.
   */
  void reset() {
    index_head = std::vector<size_t>(d - 1, 0);
    index_head_sum = 0;
    is_valid = true;
  }

  /**
   * Returns the dimension of the multi-indices (including the last-dimension index).
   */
  size_t dim() const { return d; }

  /**
   * Returns the maximum value that the first dimension of the index can take, plus one.
   */
  size_t firstIndexBound() const { return bound + 1; }

  /**
   * Returns the maximum values that the dimensions can take, plus one.
   */
  std::vector<size_t> indexBounds() const { return std::vector<size_t>(d, bound + 1); }

  /**
   * Returns the total number of multi-indices that are valid (including those with last dimension
   * != 0).
   */
  size_t numValues() const { return binom(bound + d, d); }

  /**
   * For any value 0 <= i_1, i_1 < firstIndexBound(), returns how many valid multi-indices there are
   * that start with i_1.
   */
  std::vector<size_t> numValuesPerFirstIndex() const {
    std::vector<size_t> result;
    for (size_t firstIndex = 0; firstIndex <= bound; ++firstIndex) {
      result.push_back(binom((bound - firstIndex) + (d - 1), d - 1));
    }
    return result;
  }

  /**
   * Moves the iterator directly to the position that is reached after the iteration is finished
   * (with valid() == false). This is helpful for implementing MultiDimVector.end().
   */
  void goToEnd() {
    index_head = std::vector<size_t>(d - 1, 0);
    index_head_sum = 0;
    is_valid = false;
  }

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

/**
 * Takes an iterator that does not iterate over the last index (such as a BoundedSumIterator) and
 * provides an interface to also iterate over the last index. This is used internally in
 * MultiDimVector, but can also used for other purposes.
 */
template <typename JumpIt>
class StepIterator {
  JumpIt jump_it;
  size_t first_dim_value;
  size_t last_dim_value;
  size_t last_dim_count;
  size_t tail_dims_counter;

 public:
  /**
   * Initializes to the multi-index (0, ..., 0).
   * JumpIt is the iterator object that does not iterate over the last index.
   */
  StepIterator(JumpIt jump_it)
      : jump_it(jump_it),
        first_dim_value(0),
        last_dim_value(0),
        last_dim_count(jump_it.lastDimensionCount()),
        tail_dims_counter(0){};

  /**
   * Moves to the next multi-index in lexicographical order.
   */
  void next() {
    tail_dims_counter += 1;
    last_dim_value += 1;
    if (last_dim_value >= last_dim_count) {
      last_dim_value = 0;
      jump_it.next();
      if (jump_it.firstIndex() != first_dim_value) {
        first_dim_value = jump_it.firstIndex();
        tail_dims_counter = 0;
      }
      last_dim_count = jump_it.lastDimensionCount();
    }
  }

  /**
   * Returns true if the iterator points to a valid multi-index.
   */
  bool valid() const { return jump_it.valid(); }

  /**
   * Resets the iterator to its initial state.
   */
  void reset() {
    jump_it.reset();
    first_dim_value = 0;
    last_dim_value = 0;
    last_dim_count = jump_it.lastDimensionCount();
    tail_dims_counter = 0;
  }

  /**
   * Returns the value i_1 of the current multi-index (i_1, ..., i_d).
   */
  size_t firstIndex() const { return first_dim_value; }

  /**
   * Returns n, such that the current multi-index is the (n+1)-th multi-index that starts with
   * firstIndex().
   */
  size_t tailDimsCounter() const { return tail_dims_counter; }

  /**
   * Returns a std::vector<size_t> containing the multi-index.
   */
  std::vector<size_t> index() const {
    std::vector<size_t> result = jump_it.getIndexHead();
    result.push_back(last_dim_value);
    return result;
  }
};
}
