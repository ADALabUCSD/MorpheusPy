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
from sklearn.base import BaseEstimator, ClusterMixin

class NormalizedKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, iterations=20, center_num=5):
        self.center_num = center_num
        self.iterations = iterations

    def fit(self, X, k_center):
        self.k_center, self.ya = self.k_means(X, self.iterations, self.center_num, k_center, X.shape[0])
        return self

    def k_means(self, data, iterations, center_num, k_center, rows):
        all_one = np.matrix([1] * rows).T
        all_one_k = np.matrix([1] * center_num)
        all_one_c = np.matrix([1] * k_center.shape[0]).T
        if sp.issparse(data):
            t2 = (data.power(2)).sum(axis=1).dot(all_one_k)
        else:
            t2 = (np.power(data, 2)).sum(axis=1).reshape((-1, 1)) * all_one_k
        t22 = data * 2
        ya = None

        for _ in range(iterations):
            dist = t2 - t22 * k_center + all_one * np.power(k_center, 2).sum(axis=0)
            ya = (dist == (np.amin(dist) * all_one_k))
            k_center = (data.T * ya) / (all_one_c * ya.sum(axis=0))

        return k_center, ya

