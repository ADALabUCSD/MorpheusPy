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

from morpheus.normalized_matrix import NormalizedMatrix
from scipy.sparse import csr_matrix
import numpy as np

def prepare_k(kfkds):
    result = []
    for k in kfkds:
        ns, nr = len(k), max(k) + 1
        result.append((k, csr_matrix(([1 for _ in range(ns)], ([i for i in range(ns)], k)), shape=(ns, nr))))

    return result

def normalization(m, axis=None):
    """
    Perform normalization on normalized matrix.

    :param m: normalized matrix
    :param axis: optional, if axis is 0, it will be normalized based on columns
    :return: normalized matrix
    """
    if not isinstance(m, NormalizedMatrix):
        return NotImplemented

    if axis is None or axis == 0:
        return mean_centering(m, axis=axis) / (m.max(axis=axis) - m.min(axis=axis))

    return NotImplemented


def standardization(m, axis=None):
    """
    Perform standardization on normalized matrix.

    :param m: normalized matrix
    :param axis: optional, if axis is 0, it will be standardized based on columns
    :return: normalized matrix
    """
    if not isinstance(m, NormalizedMatrix):
        return NotImplemented

    if axis is None or axis == 0:
        return mean_centering(m, axis=axis) / m.std(axis=axis)

    return NotImplemented


def mean_centering(m, axis=None):
    """
    Perform mean centering on normalized matrix.

    :param m: normalized matrix
    :param axis: optional, if axis is 0, mean will be calculated based on columns
    :return: normalized matrix
    """
    if not isinstance(m, NormalizedMatrix):
        return NotImplemented

    if axis is None or axis == 0:
        return m - m.mean(axis=axis)

    return NotImplemented


def imputation(m, axis=None, dtype=None):
    """
    Impute missing values

    :param m: normalized matrix
    :param axis:optional, if axis is 0, mean will be calculated based on columns
    :return: normalized matrix
    """
    if not isinstance(m, NormalizedMatrix):
        return NotImplemented

    if axis is None:
        mean = (np.nansum(m.ent_table) + sum([np.nansum(t[m.kfkds[i]]) for i, t in enumerate(m.att_table)])) \
               / (m.shape[0] * m.shape[1] - np.count_nonzero(np.isnan(m.ent_table)) -
                  sum([np.count_nonzero(np.isnan(t[m.kfkds[i]])) for i, t in enumerate(m.att_table)]))
        if m.ent_table.shape[0] > 0:
            m.ent_table[np.isnan(m.ent_table)] = mean
        for i in range(len(m.kfkds)):
            m.att_table[i][np.isnan(m.att_table[i])] = mean
    if axis == 0:
        if m.ent_table.shape[0] > 0:
            inds = np.where(np.isnan(m.ent_table))
            m.ent_table[inds] = np.take(np.nanmean(m.ent_table, axis=0), inds[1])
        for i in range(len(m.kfkds)):
            inds = np.where(np.isnan(m.att_table[i]))
            m.att_table[i][inds] = np.take(np.nanmean(m.att_table[i][m.kfkds[i]], axis=0), inds[1])
    return m
