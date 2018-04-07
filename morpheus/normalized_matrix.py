# Copyright 2018 Side Li and Arun Kumar
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.sparse as sp
from numpy.core.numeric import isscalar
from numpy.matrixlib.defmatrix import asmatrix, matrix
import numpy.core.numeric as N
import time
import comp

class NormalizedMatrix(matrix):
    __array_priority__ = 12.0

    def __new__(cls, ent_table, att_table, kfkds,
                dtype=None, copy=True, trans=False, stamp=None):
        """
        Matrix constructor
        
        Parameters
        ---------
        ent_table: numpy matrix
        att_table: list of numpy matrix
        kfkds: list of numpy array
        dtype: data type
        copy: whether to copy
        trans: boolean, indicating whether the matrix is transposed
        stamp: time stamp on normalized matrix

        Examples
        --------
        Entity Table:
            matrix([[ 1.  2.]
                    [ 4.  3.]
                    [ 5.  6.]
                    [ 8.  7.]
                    [ 9.  1.]])
        List of Attribute Table:
            [
                matrix([[ 1.1  2.2]
                        [ 3.3  4.4]])
            ]
        K:
            [
                array([0, 1, 1, 0, 1])
             ]
        Transposed:
            False
        """
        sizes = [ent_table.shape[1]] + [t.shape[1] for t in att_table]

        # obj = N.ndarray.__new__(NormalizedMatrix,
        #                         (sum(sizes), len(kfkds[0])) if trans else (len(kfkds[0]), sum(sizes)), dtype)
        obj = N.ndarray.__new__(NormalizedMatrix, dtype)
        obj.nshape = (sum(sizes), len(kfkds[0])) if trans else (len(kfkds[0]), sum(sizes))
        obj.ent_table = ent_table
        obj.att_table = att_table
        obj.kfkds = kfkds
        obj.trans = trans
        obj.stamp = time.clock() if stamp is None else stamp
        obj.sizes = sizes
        # used for future operators
        obj.indexes = reduce(lambda x, y: x + [(x[-1][1], x[-1][1] + y)], sizes, [(0, 0)])[2:]
        return obj

    def _copy(self, ent_table, att_table):
        """
        Copy constructor
        """
        return NormalizedMatrix(ent_table, att_table,
                                self.kfkds, dtype=self.dtype, trans=self.trans)

    def __getitem__(self, index):
        """
        Slicing is not supported. It will cause significant performance penalty on
        normalized matrix.

        :param index: slicing index
        :return: error
        """
        return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "\n".join(["Entity Table:", self.ent_table.__str__(),
              "Attribute Table:", "\n".join((t.__str__() for t in self.att_table)),
              "K matrix:", "\n".join((t.__str__() for t in self.kfkds)),
              "Transposed:", self.trans.__str__()])

    """
    Array functions are created to follow numpy semantics.
    https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
    
    1. Explicit constructor call
    2. View casting
    3. Creating from new template
    """
    def __array_prepare__(self, obj, context=None):
        pass

    def __array_wrap__(self, out_arr, context=None):
        pass

    def __array_finalize__(self, obj):
        pass

    _SUPPORTED_UFUNCS = {np.add: {1: "__add__", -1: "__radd__"},
                         np.subtract: {1: "__sub__", -1: "__rsub__"},
                         np.divide: {1: "__div__", -1: "__rdiv__"},
                         np.multiply: {1: "__mul__", -1: "__rmul__"},
                         np.power: {1: "__pow__", -1: "__rpow__"}}

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle ufunc supported in numpy standard library.
        reference: https://docs.scipy.org/doc/numpy-1.13.0/reference/ufuncs.html

        :param ufunc: ufunc object
        :param method: type of method. In this class, only __call__ is handled
        :param inputs:
        :param kwargs:
        :return: Normalized matrix or matrix or ndarray or numeric
        """
        if ufunc in self._SUPPORTED_UFUNCS and len(inputs) == 2 and method == "__call__":
            order = isinstance(inputs[0], NormalizedMatrix) - isinstance(inputs[1], NormalizedMatrix)
            if order == 1:
                return getattr(inputs[0], self._SUPPORTED_UFUNCS[ufunc][order])(inputs[1], **kwargs)
            if order == -1:
                return getattr(inputs[1], self._SUPPORTED_UFUNCS[ufunc][order])(inputs[0], **kwargs)
            if order == 0 and ufunc is np.multiply:
                return inputs[0].__mul__(inputs[1], **kwargs)

        return NotImplemented

    # Element-wise Scalar Operators
    def __add__(self, other):
        if isscalar(other):
            return self._copy(self.ent_table + other, [t + other for t in self.att_table])

        if isinstance(other, (N.ndarray, list, tuple)):
            other = asmatrix(other)
            if other.shape[1] == self.shape[1] and other.shape[0] == 1:
                return self._copy(self.ent_table + other[:, :self.ent_table.shape[1]],
                                  [t + other[:, self.indexes[i][0]:self.indexes[i][1]]
                                   for i, t in enumerate(self.att_table)])

        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isscalar(other):
            self.ent_table = self.ent_table + other
            self.att_table = [t + other for t in self.att_table]
            return self

        if isinstance(other, (N.ndarray, list, tuple)):
            other = asmatrix(other)
            if other.shape[1] == self.shape[1] and other.shape[0] == 1:
                self.ent_table = self.ent_table + other[:, :self.ent_table.shape[1]]
                self.att_table = [t + other[:, self.indexes[i][0]:self.indexes[i][1]]
                                   for i, t in enumerate(self.att_table)]
                return self

        return NotImplemented

    def __sub__(self, other):
        if isscalar(other):
            return self._copy(self.ent_table - other, [t - other for t in self.att_table])

        if isinstance(other, (N.ndarray, list, tuple)):
            other = asmatrix(other)
            if other.shape[1] == self.shape[1] and other.shape[0] == 1:
                return self._copy(self.ent_table - other[:, :self.ent_table.shape[1]],
                                  [t - other[:, self.indexes[i][0]:self.indexes[i][1]]
                                   for i, t in enumerate(self.att_table)])

        return NotImplemented

    def __rsub__(self, other):
        if isscalar(other):
            return self._copy(other - self.ent_table,
                          [other - t for t in self.att_table])

        if isinstance(other, (N.ndarray, list, tuple)):
            other = asmatrix(other)
            if other.shape[1] == self.shape[1] and other.shape[0] == 1:
                return self._copy(other[:, :self.ent_table.shape[1]] - self.ent_table,
                                  [other[:, self.indexes[i][0]:self.indexes[i][1]] - t
                                   for i, t in enumerate(self.att_table)])

        return NotImplemented

    def __isub__(self, other):
        if isscalar(other):
            self.ent_table = self.ent_table - other
            self.att_table = [t - other for t in self.att_table]
            return self

        if isinstance(other, (N.ndarray, list, tuple)):
            other = asmatrix(other)
            if other.shape[1] == self.shape[1] and other.shape[0] == 1:
                self.ent_table = self.ent_table - other[:, :self.ent_table.shape[1]]
                self.att_table = [t - other[:, self.indexes[i][0]:self.indexes[i][1]]
                                   for i, t in enumerate(self.att_table)]
                return self

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, NormalizedMatrix):
            if self.stamp == other.stamp and self.trans ^ other.trans:
                return self._cross_prod()
            else:
                return NotImplemented

        if isinstance(other, (N.ndarray, list, tuple)):
            # This promotes 1-D vectors to row vectors
            if self.trans:
                return self._right_matrix_multiplication(self, asmatrix(other).T).T
            else:
                return self._left_matrix_multiplication(self, asmatrix(other))

        if isscalar(other) or not hasattr(other, '__rmul__'):
            return self._copy(self.ent_table * other, [t * other for t in self.att_table])

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (N.ndarray, list, tuple)):
            if self.trans:
                return self._left_matrix_multiplication(self, asmatrix(other).T).T
            else:
                return self._right_matrix_multiplication(self, asmatrix(other))

        if isscalar(other) or not hasattr(other, '__rmul__'):
            return self._copy(self.ent_table * other, [t * other for t in self.att_table])

        return NotImplemented

    def __imul__(self, other):
        if not isscalar(other):
            return NotImplemented

        self.ent_table = self.ent_table * other
        self.att_table = [t * other for t in self.att_table]

        return self

    def __div__(self, other):
        if isscalar(other):
            return self._copy(self.ent_table / other, [t / other for t in self.att_table])

        if isinstance(other, (N.ndarray, list, tuple)):
            other = asmatrix(other)
            if other.shape[1] == self.shape[1] and other.shape[0] == 1:
                return self._copy(self.ent_table / other[:, :self.ent_table.shape[1]],
                                  [t / other[:, self.indexes[i][0]:self.indexes[i][1]]
                                   for i, t in enumerate(self.att_table)])

        return NotImplemented

    def __rdiv__(self, other):
        if isscalar(other):
            return self._copy(other / self.ent_table, [other / t for t in self.att_table])

        if isinstance(other, (N.ndarray, list, tuple)):
            other = asmatrix(other)
            if other.shape[1] == self.shape[1] and other.shape[0] == 1:
                return self._copy(other[:, :self.ent_table.shape[1]] / self.ent_table,
                                  [other[:, self.indexes[i][0]:self.indexes[i][1]] / t
                                   for i, t in enumerate(self.att_table)])

        return NotImplemented

    def __idiv__(self, other):
        if isscalar(other):
            self.ent_table = self.ent_table / other
            self.att_table = [t / other for t in self.att_table]
            return self

        if isinstance(other, (N.ndarray, list, tuple)):
            other = asmatrix(other)
            if other.shape[1] == self.shape[1] and other.shape[0] == 1:
                self.ent_table = self.ent_table / other[:, :self.ent_table.shape[1]]
                self.att_table = [t / other[:, self.indexes[i][0]:self.indexes[i][1]]
                                   for i, t in enumerate(self.att_table)]
                return self

        return NotImplemented

    def __pow__(self, other):
        if not isscalar(other):
            return NotImplemented

        return self._copy(
            (self.ent_table.power(other) if sp.issparse(self.ent_table) else np.power(self.ent_table, other)),
            [(t.power(other) if sp.issparse(t) else np.power(t, other)) for t in self.att_table])

    def __rpow__(self, other):
        if not isscalar(other):
            return NotImplemented

        return self._copy(np.power(other, self.ent_table.toarray() if sp.issparse(self.ent_table) else self.ent_table ),
                          [np.power(other, t.toarray() if sp.issparse(t) else t) for t in self.att_table])

    def __ipow__(self, other):
        if not isscalar(other):
            return NotImplemented
        self.ent_table = self.ent_table.power(2) if sp.issparse(self.ent_table) else np.power(self.ent_table, other)
        self.att_table = [(t.power(other) if sp.issparse(t) else np.power(t, other)) for t in self.att_table]
        return self

    # Aggregation
    def sum(self, axis=None, dtype=None, out=None):
        """
        Paramters
        ---------
        axis: None or int or tuple of ints, optional
            the axis used to perform sum aggreation.

        Examples
        --------
        T = Entity Table:
                [[ 1.  2.]
                 [ 4.  3.]
                 [ 5.  6.]
                 [ 8.  7.]
                 [ 9.  1.]]
            Attribute Table:
                [[ 1.1  2.2]
                 [ 3.3  4.4]]
            K:
                [[1, 0, 0, 1, 0]]
        >>> T.sum(axis=0)
            [[ 27.   19.   12.1  17.6]]
        >>> T.sum(axis=1)
            [[  6.3]
             [ 14.7]
             [ 18.7]
             [ 18.3]
             [ 17.7]]
        >>> T.sum()
            75.7
        """
        k = self.kfkds
        ns = k[0].shape[0]
        nr = [t.shape[0] for t in self.att_table]
        if axis == 0:
            # col sum
            if self.trans:
                return (self.ent_table.sum(axis=1) +
                        sum((t.sum(axis=1)[self.kfkds[i]] for i, t in enumerate(self.att_table)))).T
            else:
                other = np.ones((1, ns))

                res = [np.zeros((1, t.shape[0]), dtype=float) for t in self.att_table]
                comp.group(ns, len(k), 1, k, nr, other, res)

                return np.hstack(
                    [self.ent_table.sum(axis=0)] +
                    [res[i] * t for i, t in enumerate(self.att_table)])
        elif axis == 1:
            # row sum
            if self.trans:
                other = np.ones((1, ns))

                res = [np.zeros((1, t.shape[0]), dtype=float) for t in self.att_table]
                comp.group(ns, len(k), 1, k, nr, other, res)

                return np.hstack(
                    [self.ent_table.sum(axis=0)] +
                    [res * t for i, t in enumerate(self.att_table)]).T
            else:
                return self.ent_table.sum(axis=1) + \
                       sum((t.sum(axis=1)[self.kfkds[i]] for i, t in enumerate(self.att_table)))

        # sum of the whole matrix
        # res is k * r
        other = np.ones((1, ns))
        res = [np.zeros((1, t.shape[0]), dtype=float) for t in self.att_table]
        comp.group(ns, len(k), 1, k, nr, other, res)
        return self.ent_table.sum() + \
               sum((res[i] * t.sum(axis=1) for i, t in enumerate(self.att_table)))._collapse(None)

    # Multiplication
    def _left_matrix_multiplication(self, n_matrix, other):
        s = n_matrix.ent_table
        k = n_matrix.kfkds
        r = n_matrix.att_table

        ns = k[0].shape[0]
        ds = s.shape[1]
        dw = other.shape[1]

        if s.size > 0:
            res = np.asfortranarray(s.dot(other[0:ds]))
        else:
            res = np.zeros((ns, dw), dtype=float, order='C')

        v_list = []
        start = ds
        for i in range(len(k)):
            r_buff = r[i]
            nr, dr = r_buff.shape[0], r_buff.shape[1]
            end = start + dr
            v = r_buff.dot(other[start:end])
            start += dr

            v_list.append(np.asfortranarray(v))

        comp.add_new(ns, len(k), dw, k, v_list, [v.shape[0] for v in v_list], res)

        return res

    def _right_matrix_multiplication(self, n_matrix, other):
        other = other.astype(np.float, order='C')
        s = n_matrix.ent_table
        k = n_matrix.kfkds
        r = n_matrix.att_table

        ns = k[0].shape[0]
        nr = [t.shape[0] for t in r]
        nk = len(k)
        nw = other.shape[0]
        res = [np.zeros((nw, t.shape[0]), dtype=float) for t in r]

        comp.group(ns, nk, nw, k, nr, other, res)

        return np.hstack(([other * s if sp.issparse(s) else other.dot(s)] if s.shape[1] != 0 else []) +
                         [(res[i] * t if sp.issparse(t) else res[i].dot(t)) for i, t in enumerate(r)])

    def _cross_prod(self):
        s = self.ent_table
        r = self.att_table
        k = self.kfkds
        ns = k[0].shape[0]
        ds = s.shape[1]
        nr = [t.shape[0] for t in self.att_table]
        if not self.trans:
            if all(map(sp.issparse, self.att_table)):
                # TODO
                return NotImplemented
                # return self._t_cross(self.ent_table) + \
                #        sum((self._t_cross(self._t_cross(t)[self.kfkds[i][0]], self.kfkds[i][1]))
                #            for i, t in enumerate(self.att_table))
            else:
                if s.size > 0:
                    res = self._t_cross(s)
                else:
                    res = np.zeros((ns, ns), dtype=float, order='C')

                cross_r = [self._t_cross(t) for t in r]
                comp.expand_add(ns, len(k), k, cross_r, nr, res)

                return res

        else:
            if all(map(sp.issparse, self.att_table)):
                other = np.ones((1, ns))
                v = [np.zeros((1, t.shape[0]), dtype=float) for t in self.att_table]
                comp.group(ns, len(k), 1, k, nr, other, v)
                size = self.att_table[0].size
                data = np.empty(size)

                # part 2 and 3 are p.T and p
                comp.multiply_sparse(size, self.att_table[0].row, self.att_table[0].data, np.sqrt(v[0]), data)
                diag_part = self._cross(sp.coo_matrix((data, (self.att_table[0].row, self.att_table[0].col))))
                if ds > 0:
                    m = np.zeros((nr[0], ds))
                    comp.group_left(ns, ds, s, k[0], m)
                    p = self._cross(self.att_table[0], m)
                    s_part = self._cross(self.ent_table)

                    res = sp.vstack((sp.hstack((s_part, p.T)), sp.hstack((p, diag_part))))
                else:
                    res = diag_part
            else:
                nt = self.ent_table.shape[1] + self.att_table[0].shape[1]
                other = np.ones((1, ns))
                v = [np.zeros((1, t.shape[0]), dtype=float) for t in self.att_table]
                data = np.empty(self.att_table[0].shape, order='C')

                res = np.empty((nt, nt))
                comp.group(ns, len(k), 1, k, nr, other, v)
                comp.multiply(self.att_table[0].shape[0], self.att_table[0].shape[1], self.att_table[0], v[0], data)

                res[ds:, ds:] = self._cross(data)
                if ds > 0:
                    m = np.zeros((nr[0], ds))
                    comp.group_left(ns, ds, s, k[0], m)
                    res[ds:, :ds] = self._cross(self.att_table[0], m)
                    res[:ds, ds:] = res[ds:, :ds].T
                    res[:ds, :ds] = self._cross(self.ent_table)

            # TODO: m:n joins
            # for i in range(1, len(self.kfkds)):
            #     p = self._cross(self.att_table[i], self._cross(self.kfkds[i][1], self.ent_table))
            #     for j in range(i - 1):
            #         p = np.hstack((p, np.cross(self.att_table[i], self._cross(
            #             self._cross(self.kfkds[i][1], self.kfkds[i][1]), self.att_table[j]))))
            #     res = np.vstack((np.hstack((res, p.T)),
            #                      np.hstack((p, self._cross(np.diag(
            #                          np.power(np.diagflat(self.kfkds[i][1].sum(axis=0)), 0.5) * self.att_table[i]))))))

            return res

    def _t_cross(self, matrix_a, matrix_b=None):
        if sp.issparse(matrix_a) or sp.issparse(matrix_b):
            if matrix_b is None:
                return matrix_a * matrix_a.T
            else:
                return matrix_a * matrix_b.T
        else:
            if matrix_b is None:
                return matrix_a.dot(matrix_a.T)
            else:
                return matrix_a.dot(matrix_b.T)

    def _cross(self, matrix_a, matrix_b=None):
        if sp.issparse(matrix_a) or sp.issparse(matrix_b):
            if matrix_b is None:
                return matrix_a.T * (matrix_a)
            else:
                return matrix_a.T * (matrix_b)
        else:
            if matrix_b is None:
                return matrix_a.T.dot(matrix_a)
            else:
                return matrix_a.T.dot(matrix_b)

    def dot(self, other):
        return self.__mul__(other)

    def max(self, axis=None, out=None):
        """
        Calculate the maximum element per table or per column.
        Signatures are the same as numpy matrix to ensure downstream compatibility.

        :param axis: optional, only column wise (axis=0) operation is supported.
        :param out:
        :return: numpy matrix or numeric
        """
        if axis is None:
            return max(self.ent_table.max(), max(t.max() for t in self.att_table))

        if axis == 0:
            return np.hstack((self.ent_table.max(axis=0),
                             np.hstack([t.max(axis=0) for t in self.att_table])))

        return NotImplemented

    def min(self, axis=None, out=None):
        """
        Calculate the minmum element per table or per column.
        Signatures are the same as numpy matrix to ensure downstream compatibility.

        :param axis: optional, only column wise (axis=0) operation is supported.
        :param out:
        :return: numpy matrix or numeric
        """
        if axis is None:
            return min(self.ent_table.min(), max(t.min() for t in self.att_table))

        if axis == 0:
            return np.hstack((self.ent_table.min(axis=0),
                             np.hstack([t.min(axis=0) for t in self.att_table])))

        return NotImplemented

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        """
        Calculate the variance per table or per column.
        Signatures are the same as numpy matrix to ensure downstream compatibility.

        :param axis: optional, only column wise (axis=0) operation is supported.
        :param dtype: data type
        :param ddof:
        :param out:
        :return: numpy matrix or numeric
        """
        if axis is None:
            return ((self - self.mean()) ** 2).mean()

        if axis == 0:
            return np.hstack([self.ent_table.var(axis=0)] +
                             [self.att_table[i][self.kfkds[i]].var(axis=0) for i in range(len(self.kfkds))])

        return NotImplemented

    def std(self, axis=None, dtype=None, out=None, ddof=0):
        """
        Calculate the standard deviation per table or per column.
        Signatures are the same as numpy matrix to ensure downstream compatibility.

        :param axis: optional, only column wise (axis=0) operation is supported.
        :param dtype: data type
        :param out:
        :param ddof:
        :return: numpy matrix or numeric
        """
        if axis is None:
            return np.sqrt(self.var())

        if axis == 0:
            return np.sqrt(self.var(axis=0))

        return NotImplemented

    def mean(self, axis=None, dtype=None, out=None):
        """
        Calculate the mean per table or per column.
        Signatures are the same as numpy matrix to ensure downstream compatibility.

        :param axis: optional, only column wise (axis=0) operation is supported.
        :param dtype: data type
        :param out:
        :return: numpy matrix or numeric
        """
        if axis is None:
            return self.sum() / (self.shape[0] * self.shape[1])

        if axis == 0:
            return np.hstack([self.ent_table.mean(axis=0)] +
                             [self.att_table[i][self.kfkds[i]].mean(axis=0) for i in range(len(self.kfkds))])


        return NotImplemented

    def transpose(self):
        return NormalizedMatrix(self.ent_table, self.att_table, self.kfkds,
                                dtype=self.dtype, trans=(not self.trans), stamp=self.stamp)

    @property
    def I(self):
        if self.trans:
            return self * self._cross_prod().I
        else:
            if self.shape[0] > self.shape[1]:
                return np.mat(self.T * self).I * self.T
            else:
                return self.T * np.mat(self * self.T).I
    @property
    def T(self):
        return self.transpose()

    @property
    def shape(self):
        return self.nshape
