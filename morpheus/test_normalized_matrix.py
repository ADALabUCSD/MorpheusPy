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
import sklearn.preprocessing as preprocess
from numpy.testing import (
    run_module_suite, assert_equal, assert_almost_equal
)

import normalized_matrix as nm
import utils


class TestNormalizedMatrix(object):
    s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
    k = [np.array([0, 1, 1, 0, 1])]

    r = [np.matrix([[1.1, 2.2], [3.3, 4.4]])]
    m = np.matrix([[1.0, 2.0, 1.1, 2.2],
                   [4.0, 3.0, 3.3, 4.4],
                   [5.0, 6.0, 3.3, 4.4],
                   [8.0, 7.0, 1.1, 2.2],
                   [9.0, 1.0, 3.3, 4.4]])
    n_matrix = nm.NormalizedMatrix(s, r, k)

    def test_add(self):
        n_matrix = self.n_matrix

        local_matrix = n_matrix + 1
        assert_equal(local_matrix.ent_table, self.s + 1)
        assert_equal(local_matrix.att_table[0], self.r[0] + 1)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = n_matrix + np.matrix([1.0, 2.0, 1.1, 2.2])
        assert_equal(local_matrix.ent_table, self.s + np.matrix([1.0, 2.0]))
        assert_equal(local_matrix.att_table[0], self.r[0] + np.matrix([1.1, 2.2]))
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = 1 + n_matrix
        assert_equal(local_matrix.ent_table[0], 1 + self.s[0])
        assert_equal(local_matrix.att_table[0], 1 + self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.matrix([1.0, 2.0, 1.1, 2.2]) + n_matrix
        assert_equal(local_matrix.ent_table, np.matrix([1.0, 2.0]) + self.s)
        assert_equal(local_matrix.att_table[0], np.matrix([1.1, 2.2]) + self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = n_matrix + 1
        local_matrix += 1
        assert_equal(local_matrix.ent_table[0], 2 + self.s[0])
        assert_equal(local_matrix.att_table[0], 2 + self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = n_matrix + np.matrix([1.0, 2.0, 1.1, 2.2])
        local_matrix += np.matrix([1.0, 2.0, 1.1, 2.2])

        assert_equal(local_matrix.ent_table, self.s + np.matrix([1.0, 2.0]) * 2)
        assert_equal(local_matrix.att_table[0], self.r[0] + np.matrix([1.1, 2.2]) * 2)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.add(n_matrix, 1)
        assert_equal(local_matrix.ent_table, self.s + 1)
        assert_equal(local_matrix.att_table[0], self.r[0] + 1)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.add(1, n_matrix)
        assert_equal(local_matrix.ent_table, self.s + 1)
        assert_equal(local_matrix.att_table[0], self.r[0] + 1)
        assert_equal(local_matrix.kfkds[0], self.k[0])

    def test_sub(self):
        n_matrix = self.n_matrix

        local_matrix = n_matrix - 1
        assert_equal(local_matrix.ent_table[0], self.s[0] - 1)
        assert_equal(local_matrix.att_table[0], self.r[0] - 1)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = n_matrix - np.matrix([1.0, 2.0, 1.1, 2.2])
        assert_equal(local_matrix.ent_table, self.s - np.matrix([1.0, 2.0]))
        assert_equal(local_matrix.att_table[0], self.r[0] - np.matrix([1.1, 2.2]))
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = 1 - n_matrix
        assert_equal(local_matrix.ent_table[0], 1 - self.s[0])
        assert_equal(local_matrix.att_table[0], 1 - self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.matrix([1.0, 2.0, 1.1, 2.2]) - n_matrix
        assert_equal(local_matrix.ent_table, np.matrix([1.0, 2.0]) - self.s)
        assert_equal(local_matrix.att_table[0], np.matrix([1.1, 2.2]) - self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = n_matrix - 1
        local_matrix -= 1

        assert_equal(local_matrix.ent_table[0], self.s[0] - 2)
        assert_equal(local_matrix.att_table[0], self.r[0] - 2)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = n_matrix - np.matrix([1.0, 2.0, 1.1, 2.2])
        local_matrix -= np.matrix([1.0, 2.0, 1.1, 2.2])

        assert_equal(local_matrix.ent_table, self.s - np.matrix([1.0, 2.0]) * 2)
        assert_equal(local_matrix.att_table[0], self.r[0] - np.matrix([1.1, 2.2]) * 2)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.subtract(n_matrix, 1)
        assert_equal(local_matrix.ent_table, self.s - 1)
        assert_equal(local_matrix.att_table[0], self.r[0] - 1)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.subtract(1, n_matrix)
        assert_equal(local_matrix.ent_table, 1 - self.s)
        assert_equal(local_matrix.att_table[0], 1 - self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

    def test_mul(self):
        n_matrix = self.n_matrix

        local_matrix = n_matrix * 2
        assert_equal(local_matrix.ent_table[0], self.s[0] * 2)
        assert_equal(local_matrix.att_table[0], self.r[0] * 2)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = 2 * n_matrix
        assert_equal(local_matrix.ent_table[0], 2 * self.s[0])
        assert_equal(local_matrix.att_table[0], 2 * self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = 2 * n_matrix
        local_matrix *= 2
        assert_equal(local_matrix.ent_table[0], 4 * self.s[0])
        assert_equal(local_matrix.att_table[0], 4 * self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.multiply(n_matrix, 2)
        assert_equal(local_matrix.ent_table, self.s * 2)
        assert_equal(local_matrix.att_table[0], self.r[0] * 2)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.multiply(2, n_matrix)
        assert_equal(local_matrix.ent_table, 2 * self.s)
        assert_equal(local_matrix.att_table[0], 2 * self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

    def test_div(self):
        n_matrix = self.n_matrix

        local_matrix = n_matrix / 2
        assert_equal(local_matrix.ent_table, self.s / 2)
        assert_equal(local_matrix.att_table[0], self.r[0] / 2)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = n_matrix / np.matrix([1.0, 2.0, 1.1, 2.2])
        assert_equal(local_matrix.ent_table, self.s / np.matrix([1.0, 2.0]))
        assert_equal(local_matrix.att_table[0], self.r[0] / np.matrix([1.1, 2.2]))
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = 2 / n_matrix
        assert_equal(local_matrix.ent_table, 2 / self.s)
        assert_equal(local_matrix.att_table[0], 2 / self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.matrix([1.0, 2.0, 1.1, 2.2]) / n_matrix
        assert_equal(local_matrix.ent_table, np.matrix([1.0, 2.0]) / self.s)
        assert_equal(local_matrix.att_table[0], np.matrix([1.1, 2.2]) / self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = n_matrix / 2
        local_matrix /= 2

        assert_equal(local_matrix.ent_table, self.s / 4)
        assert_equal(local_matrix.att_table[0], self.r[0] / 4)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = n_matrix / np.matrix([1.0, 2.0, 1.1, 2.2])
        local_matrix /= np.matrix([1.0, 2.0, 1.1, 2.2])

        assert_equal(local_matrix.ent_table, self.s / np.power(np.matrix([1.0, 2.0]), 2))
        assert_equal(local_matrix.att_table[0], self.r[0] / np.power(np.matrix([1.1, 2.2]), 2))
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.divide(n_matrix, 2)
        assert_equal(local_matrix.ent_table, self.s / 2)
        assert_equal(local_matrix.att_table[0], self.r[0] / 2)
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.divide(2, n_matrix)
        assert_equal(local_matrix.ent_table, 2 / self.s)
        assert_equal(local_matrix.att_table[0], 2 / self.r[0])
        assert_equal(local_matrix.kfkds[0], self.k[0])

    def test_pow(self):
        n_matrix = self.n_matrix

        local_matrix = n_matrix ** 2
        assert_equal(local_matrix.ent_table, np.power(self.s, 2))
        assert_equal(local_matrix.att_table[0], np.power(self.r[0], 2))
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.power(n_matrix, 2)
        assert_equal(local_matrix.ent_table, np.power(self.s, 2))
        assert_equal(local_matrix.att_table[0], np.power(self.r[0], 2))
        assert_equal(local_matrix.kfkds[0], self.k[0])

        local_matrix = np.power(2, n_matrix)
        assert_equal(local_matrix.ent_table, np.power(2, self.s))
        assert_equal(local_matrix.att_table[0], np.power(2, self.r[0]))
        assert_equal(local_matrix.kfkds[0], self.k[0])

    def test_transpose(self):
        n_matrix = self.n_matrix
        assert_equal(n_matrix.T.T.sum(axis=0), n_matrix.sum(axis=0))
        assert_equal(np.array_equal(n_matrix.T.sum(axis=0), n_matrix.sum(axis=0)), False)

    def test_inverse(self):
        n_matrix = self.n_matrix

        assert_almost_equal(n_matrix.I, self.n_matrix.I)

    def test_row_sum(self):
        n_matrix = self.n_matrix

        assert_almost_equal(n_matrix.sum(axis=1), self.m.sum(axis=1))

    def test_row_sum_trans(self):
        n_matrix = nm.NormalizedMatrix(self.s, self.r, self.k, trans=True)

        assert_almost_equal(n_matrix.sum(axis=1), self.m.T.sum(axis=1))

    def test_col_sum(self):
        n_matrix = self.n_matrix

        assert_almost_equal(n_matrix.sum(axis=0), self.m.sum(axis=0))

    def test_row_col_trans(self):
        n_matrix = nm.NormalizedMatrix(self.s, self.r, self.k, trans=True)

        assert_almost_equal(n_matrix.sum(axis=0), self.m.T.sum(axis=0))

    def test_sum(self):
        n_matrix = self.n_matrix

        assert_almost_equal(n_matrix.sum(), self.m.sum())

    def test_lmm(self):
        n_matrix = self.n_matrix
        x = np.matrix([[1.0], [2.0], [3.0], [4.0]])

        assert_equal(n_matrix * x, self.m * x)

    def test_lmm_trans(self):
        n_matrix = self.n_matrix.T
        x = np.matrix([[1.0], [2.0], [3.0], [4.0], [5.0]])

        assert_almost_equal(n_matrix * x, self.m.T * x)

    def test_rmm(self):
        n_matrix = self.n_matrix
        x = np.matrix([[1.0, 2.0, 3.0, 4.0, 5.0]])

        assert_almost_equal(x * n_matrix, x * self.m)

    def test_rmm_trans(self):
        n_matrix = self.n_matrix
        x = np.matrix([[1.0, 2.0, 3.0, 4.0]])
        assert_equal(x * n_matrix.T, x * self.m.T)

    def test_cross_prod(self):
        n_matrix = self.n_matrix.T * self.n_matrix
        assert_almost_equal(n_matrix, self.m.T * self.m)

        n_matrix = np.multiply(self.n_matrix.T, self.n_matrix)
        assert_almost_equal(n_matrix, self.m.T * self.m)

        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]), np.array([0, 1, 1, 1, 0])]
        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]), np.matrix([[0.1, 0.2], [0.3, 0.4]])]
        n_matrix = nm.NormalizedMatrix(s, r, k)
        m = np.hstack([s, r[0][k[0]], r[1][k[1]]])

        assert_almost_equal(n_matrix.T * n_matrix, m.T * m)

        n_matrix = nm.NormalizedMatrix(s, [sp.coo_matrix(ri) for ri in r], k)
        assert_almost_equal((n_matrix.T * n_matrix).toarray(), m.T * m)

    def test_cross_prod_trans(self):
        n_matrix = self.n_matrix.T
        n_matrix = n_matrix.T * n_matrix
        assert_almost_equal(n_matrix, self.m * self.m.T)

        n_matrix = nm.NormalizedMatrix(self.s, [sp.coo_matrix(att) for att in self.r], self.k).T
        n_matrix = n_matrix.T * n_matrix
        assert_almost_equal(n_matrix, self.m * self.m.T)

    def test_max(self):
        n_matrix = self.n_matrix

        assert_equal(n_matrix.max(), self.m.max())
        assert_equal(n_matrix.max(axis=0), self.m.max(axis=0))

    def test_min(self):
        n_matrix = self.n_matrix

        assert_equal(n_matrix.min(), self.m.min())
        assert_equal(n_matrix.min(axis=0), self.m.min(axis=0))

    def test_mean(self):
        n_matrix = self.n_matrix

        assert_equal(n_matrix.mean(), self.m.mean())
        assert_equal(n_matrix.mean(axis=0), self.m.mean(axis=0))

    def test_var(self):
        n_matrix = self.n_matrix

        assert_equal(n_matrix.var(), self.m.var())
        assert_equal(n_matrix.var(axis=0), self.m.var(axis=0))

    def test_std(self):
        n_matrix = self.n_matrix

        assert_equal(n_matrix.std(), self.m.std())
        assert_equal(n_matrix.std(axis=0), self.m.std(axis=0))

    def test_mean_centering(self):
        n_matrix = utils.mean_centering(self.n_matrix)

        assert_equal(np.hstack((n_matrix.ent_table, n_matrix.att_table[0][n_matrix.kfkds[0]])),
                     self.m - self.m.mean())

        n_matrix = utils.mean_centering(self.n_matrix, axis=0)
        scaler = preprocess.StandardScaler(with_std=False)
        scaler.fit(self.m)
        assert_equal(np.hstack((n_matrix.ent_table, n_matrix.att_table[0][n_matrix.kfkds[0]])),
                     scaler.transform(self.m))

    def test_standardization(self):
        n_matrix = utils.standardization(self.n_matrix)
        assert_equal(np.hstack((n_matrix.ent_table, n_matrix.att_table[0][n_matrix.kfkds[0]])),
                     (self.m - self.m.mean()) / self.m.std())

        n_matrix = utils.standardization(self.n_matrix, axis=0)
        scaler = preprocess.StandardScaler()
        scaler.fit(self.m)
        assert_equal(np.hstack((n_matrix.ent_table, n_matrix.att_table[0][n_matrix.kfkds[0]])),
                     scaler.transform(self.m))

    def test_normalization(self):
        n_matrix = utils.normalization(self.n_matrix)
        assert_equal(np.hstack((n_matrix.ent_table, n_matrix.att_table[0][n_matrix.kfkds[0]])),
                     (self.m - self.m.mean()) / (self.m.max() - self.m.min()))

    def test_imputation(self):
        s = np.matrix([[1.0, np.nan], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = self.k
        r = [np.matrix([[np.nan, 2.2], [3.3, 4.4]])]
        m = np.hstack([s, r[0][k[0]]])
        m[np.isnan(m)] = np.nanmean(m)
        n_matrix = nm.NormalizedMatrix(s, r, k)

        assert_equal(utils.imputation(n_matrix).sum(axis=0), m.sum(axis=0))

        s = np.matrix([[1.0, np.nan], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]), np.array([0, 1, 1, 0, 1])]
        r = [np.matrix([[np.nan, 2.2], [3.3, 4.4]]), np.matrix([[np.nan, 2.2], [3.3, 4.4]])]
        m = np.hstack([s, r[0][k[0]], r[1][k[1]]])

        mean = np.nanmean(m, axis=0)
        inds = np.where(np.isnan(m))
        m[inds] = np.take(mean, inds[1])
        n_matrix = nm.NormalizedMatrix(s, r, k)
        assert_almost_equal(utils.imputation(n_matrix, axis=0).sum(axis=0), m.sum(axis=0))

if __name__ == "__main__":
    run_module_suite()
