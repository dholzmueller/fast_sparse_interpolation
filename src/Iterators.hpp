// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <vector>

namespace fsi {

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

template <typename JumpIt>
class StepIterator {
  JumpIt jump_it;
  size_t first_dim_value;
  size_t last_dim_value;
  size_t last_dim_count;
  size_t tail_dims_counter;

 public:
  StepIterator(JumpIt jump_it)
      : jump_it(jump_it),
        first_dim_value(0),
        last_dim_value(0),
        last_dim_count(jump_it.lastDimensionCount()),
        tail_dims_counter(0){};

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

  bool valid() const { return jump_it.valid(); }

  void reset() {
    jump_it.reset();
    first_dim_value = 0;
    last_dim_value = 0;
    last_dim_count = jump_it.lastDimensionCount();
    tail_dims_counter = 0;
  }

  size_t firstIndex() const { return first_dim_value; }
  size_t tailDimsCounter() const { return tail_dims_counter; }

  std::vector<size_t> index() const {
    std::vector<size_t> result = jump_it.getIndexHead();
    result.push_back(last_dim_value);
    return result;
  }
};

class BoundedSumIterator {
  size_t d;
  size_t bound;
  std::vector<size_t> index_head;  // contains all entries of the index except the last one
  size_t index_head_sum;
  bool is_valid;

 public:
  BoundedSumIterator(size_t d, size_t bound)
      : d(d), bound(bound), index_head(d - 1, 0), index_head_sum(0), is_valid(true){};

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
        index_head[0] = 0;
        index_head_sum = 0;
        is_valid = false;
      }
    }
  }

  size_t firstIndex() const { return index_head[0]; }

  size_t indexAt(size_t dim) const { return index_head[dim]; }

  std::vector<size_t> getIndexHead() const { return index_head; }

  bool valid() const { return is_valid; }

  void reset() {
    index_head = std::vector<size_t>(d - 1, 0);
    index_head_sum = 0;
    is_valid = true;
  }

  size_t dim() const { return d; }

  size_t firstIndexBound() const { return bound + 1; }

  std::vector<size_t> indexBounds() const { return std::vector<size_t>(d, bound + 1); }

  size_t numValues() const { return binom(bound + d, d); }

  std::vector<size_t> numValuesPerFirstIndex() const {
    std::vector<size_t> result;
    for (size_t firstIndex = 0; firstIndex <= bound; ++firstIndex) {
      result.push_back(binom((bound - firstIndex) + (d - 1), d - 1));
    }
    return result;
  }

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
}
