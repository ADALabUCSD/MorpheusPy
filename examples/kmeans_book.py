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

from morpheus.algorithms.kmeans import NormalizedKMeans

import numpy as np
from scipy.io import mmread
from scipy.sparse import hstack
import morpheus.normalized_matrix as nm

s = np.matrix([])

join_set1 = np.genfromtxt('./data/BookCrossing/MLFK1.csv', skip_header=True, dtype=int)
num_s = len(join_set1)
num_r1 = max(join_set1)
r1 = mmread('./data/BookCrossing/MLR1Sparse.txt',)

join_set2 = np.genfromtxt('./data/BookCrossing/MLFK2.csv', skip_header=True, dtype=int)
num_s = len(join_set2)
num_r2 = max(join_set2)
r2 = mmread('./data/BookCrossing/MLR2Sparse.txt',)

Y = np.matrix(np.genfromtxt('./data/BookCrossing/MLY.csv', skip_header=True, dtype=int)).T
k = [join_set1 - 1, join_set2 - 1]
T = hstack((r1.tocsr()[k[0]], r2.tocsr()[k[1]]))

iterations = 1
result_eps = 1e-6
center_number = 5
k_center = (T.toarray()[:center_number, : num_s + num_r1 + num_r2]).T

print "start factorized matrix"
normalized_matrix = nm.NormalizedMatrix(s, [r1, r2], k)
print "end factorized matrix"

import time
print "start materialized regression"
m_cluster = NormalizedKMeans()
start = time.time()
m_cluster.fit(T, k_center)
end = time.time()
print "end materialized regression"

m_time = end - start

print "start factorized regression"
n_cluster = NormalizedKMeans()
start = time.time()
n_cluster.fit(normalized_matrix, k_center)
end = time.time()
print "end factorized regression"

n_time = end - start

print "speedup is", m_time / n_time
