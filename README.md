# MorpheusPy

MopheusPy is a Python implementation of factorized learning described in paper: [Towards Linear Algebra over Normalized Data](https://adalabucsd.github.io/papers/2017_Morpheus_VLDB.pdf).

## Prerequisite
- Python 2.7
- NumPy 1.13
- SciPy 1.0.0
- SciKit Learn
- Python developer kit (C++ rewrite is used)

## Installation
- To install, simply run `python setup.py install`.
- To build C++ module, run `python setup.py build_ext --inplace`.

## Usage
Simply import `NormalizedMatrix` like a NumPy matrix. Then apply linear algebra operations on it.
E.g

```
s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
k = [np.array([0, 1, 1, 0, 1])]
r = [np.matrix([[1.1, 2.2], [3.3, 4.4]])]
x = np.matrix([[1.0], [2.0], [3.0], [4.0]])

# materialized matrix
m = np.matrix([[1.0, 2.0, 1.1, 2.2],
               [4.0, 3.0, 3.3, 4.4],
               [5.0, 6.0, 3.3, 4.4],
               [8.0, 7.0, 1.1, 2.2],
               [9.0, 1.0, 3.3, 4.4]])
# normalized matrix
n_matrix = nm.NormalizedMatrix(s, r, k)

# result:
n_matrix.dot(x) == m_matrix.dot(x)

```    
## Operators supported
add, substract, divide, multiply, dot product, cross product, inverse

## Note
This library is implemented as a high-level wrapper. It might conflict with existing machine learning libraries if the libraries skip high-level implementation to optimize linear algebra in C kernel.




