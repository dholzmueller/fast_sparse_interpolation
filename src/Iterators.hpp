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

  bool done() { return jump_it.done(); }

  void reset() {
    jump_it.reset();
    first_dim_value = 0;
    last_dim_value = 0;
    last_dim_count = jump_it.lastDimensionCount();
    tail_dims_counter = 0;
  }
};

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
}
