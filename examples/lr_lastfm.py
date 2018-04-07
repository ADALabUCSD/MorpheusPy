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

from morpheus.algorithms.logistic_regression import NormalizedLogisticRegression
import numpy as np
from scipy.io import mmread
from scipy.sparse import hstack
import morpheus.normalized_matrix as nm

s = np.matrix([])

join_set1 = np.genfromtxt('./data/LastFM/MLFK1.csv', skip_header=True, dtype=int)
r1 = mmread('./data/LastFM/MLR1Sparse.txt',)

join_set2 = np.genfromtxt('./data/LastFM/MLFK2.csv', skip_header=True, dtype=int)
r2 = mmread('./data/LastFM/MLR2Sparse.txt',)

Y = np.matrix(np.genfromtxt('./data/LastFM/MLY.csv', skip_header=True)).T
k = [join_set1 - 1, join_set2 - 1]
T = hstack((r1.tocsr()[k[0]], r2.tocsr()[k[1]]))
print T.shape

w_init = np.matrix(np.random.randn(T.shape[1], 1))
w_init2 = np.matrix(w_init, copy=True)
gamma = 0.000001
iterations = 20
result_eps = 1e-6

print "start factorized matrix"
normalized_matrix = nm.NormalizedMatrix(s, [r1, r2], k)
print "end factorized matrix"

import time
def one_run_m():
    print "start materialized regression"
    m_regressor = NormalizedLogisticRegression()
    start = time.time()
    m_regressor.fit(T, Y, w_init=w_init)
    end = time.time()
    print "end materialized regression"
    return end - start

def one_run_n():
    print "start factorized regression"
    n_regressor = NormalizedLogisticRegression()
    start = time.time()
    n_regressor.fit(normalized_matrix, Y, w_init=w_init2)
    end = time.time()
    print "end factorized regression"

    return end - start

m_time = np.mean([one_run_m() for _ in range(5)])
n_time = np.mean([one_run_n() for _ in range(5)])

print "speedup is", m_time / n_time
