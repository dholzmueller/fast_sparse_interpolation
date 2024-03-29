This is a header-only library providing a fast matrix-vector product and linear system solver for tensor product matrices with a downward-closed index set restriction. It can especially be applied to compute interpolation coefficients for a sparse grid basis or evaluate a sparse interpolant at sparse grid points. The library code is located in the src folder.

The fast_sparse_interpolation library is published under an Apache 2.0 license. If you use this project for research purposes, please cite the following publication describing the mathematical background:
- [David Holzmüller and Dirk Pflüger. Fast Sparse Grid Operations using the Unidirectional Principle: A Generalized and Unified Framework. Sparse Grids and Applications - Munich 2018 (2021).](https://link.springer.com/chapter/10.1007/978-3-030-81362-8_4)

The essential ideas behind the algorithm were first proposed in:
- Gustavo Avila and Tucker Carrington Jr. A multi-dimensional Smolyak collocation method in curvilinear coordinates for computing vibrational spectra (2015).

Example code for usage can be found in test/main.cpp. This file can be compiled by executing the top-level Makefile (make debug / make release). This Makefile executes cmake on CMakeLists.txt to generate a Makefile in the build folder, which is then automatically executed using make to compile the executable test code. The test folder also contains code for performance measurement and plotting of performance data, which we used for our paper.

The C++ code requires the `boost` library and the python code requires the `numpy` and `matplotlib` libraries. The code has been tested on Ubuntu 20.10 with gcc 10.2.0, cmake 3.16.3, boost 1.71.0, Python 3.8.6, numpy 1.18.5 and matplotlib 3.2.2. It should be relative flexible with respect to software versions and operating systems, however. The resulting test and plotting code can be executed from the command line using the top-level folder as a working directory via `./build/fsi-test`, `python3 test/tex_line_plot.py` and `python3 test/surf_plot.py`.

All functions and classes lie inside the namespace fsi (short for fast_sparse_interpolation). The basic structure is as follows:
- The class MultiDimVector<IteratorType> stores vectors indexed by multi-indices.
- The class SparseTPOperator<IteratorType> implicitly stores matrices of a certain type (restrictions of tensor product matrices to a downward closed index set), as described in the paper. It implements a matrix-vector product (apply()) and an inverse-matrix-vector product (solve()).
- The class BoundedSumIterator provides an iterator that allows to iterate over a downward closed index set, specifically a set of the form {(i_1, ..., i_d) | i_1, ..., i_d >= 0, i_1 + ... + i_d <= bound}. It is possible to provide other iterator classes with the same interface in order to use different downward closed index sets.
- The function evaluateFunction() evaluates a function at grid points and returns a MultiDimVector with the function values.
- The function createInterpolationOperator() creates a SparseTPOperator whose matrix corresponds to a sparse grid interpolation problem with given basis functions and grid points.
- For custom extensions (such as evaluating derivatives), helper functions for multiplication with matrices of the form ($\widehat{I \otimes \hdots \otimes I \otimes M}$) are available (for arbitrary M, for lower triangular M, for upper triangular M and for M = I). Each single multiplication permutes the indices in the MultiDimVector, after d multiplications (where d is the number of dimensions), the indices are ordered as in the beginning.

The basis functions and grid points used in the examples are only easy-to-implement examples. For practical purposes, you should probably use other basis functions and grid points, for example ones provided in the SG++ library. Recommendations:
- Legendre polynomials and Leja points (or L2-Leja points, see SG++)
- Splines and uniform points
- Maybe also kernel functions or trigonometric functions and uniform points
