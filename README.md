This is a header-only library providing a fast matrix-vector product and linear system solver for tensor product matrices with a downward-closed index set restriction. This can especially be applied to compute interpolation coefficients for a sparse grid basis or evaluate a sparse interpolant at sparse grid points.

The fast_sparse_interpolation library is published under an Apache 2.0 license. If you use this project for research purposes, please cite one of the following publications:
- David Holzmüller, Dirk Pflüger: Fast Sparse Grid Interpolation Coefficients Via LU Decomposition (2019).

Example code for usage can be found in test/main.cpp. This file can be compiled by executing the top-level Makefile (make debug / make release). This Makefile executes cmake on CMakeLists.txt to generate a Makefile in the build folder, which is then automatically executed using make to compile the library. 


Note for developers:
To run the script ./prepend_header.sh, modify it to only prepend the header in files that do not contain one and run
shopt -s globstar
beforehand


