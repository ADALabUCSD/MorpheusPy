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
from scipy.io import mmread
from scipy.sparse import hstack

import morpheus.normalized_matrix as nm
from morpheus.algorithms.GNMF import GaussianNMF

s = np.matrix(np.genfromtxt('./data/Walmart/MLSraw.txt', skip_header=True, dtype=int)).T

join_set1 = np.genfromtxt('./data/Walmart/MLFK1.csv', skip_header=True, dtype=int)
r1 = mmread('./data/Walmart/MLR1Sparse.txt')

join_set2 = np.genfromtxt('./data/Walmart/MLFK2.csv', skip_header=True, dtype=int)
r2 = mmread('./data/Walmart/MLR2Sparse.txt')

k = [join_set1 - 1, join_set2 - 1]
T = hstack((s, r1.tocsr()[k[0]], r2.tocsr()[k[1]]))
Y = np.matrix(np.genfromtxt('./data/Walmart/MLY.csv', skip_header=True, dtype=int)).T

w_init = np.mat(np.random.rand(T.shape[0], 5))
h_init = np.mat(np.random.rand(5, T.shape[1]))
iterations = 20

print "start factorized matrix"
normalized_matrix = nm.NormalizedMatrix(s, [r1, r2], k)
print "end factorized matrix"

import time
m = GaussianNMF()
print "start materialized NMF"
start = time.time()
m.fit(T, w_init=w_init, h_init=h_init)
end = time.time()
print "end materialized NMF"

m_time = end - start

print "start factorized NMF"
n = GaussianNMF()
start = time.time()
n.fit(normalized_matrix, w_init=w_init, h_init=h_init)
end = time.time()
print "end factorized NMF"

n_time = end - start

print "speedup is ", m_time / n_time
