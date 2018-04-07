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
import time

import morpheus.normalized_matrix as nm
import plotly
import plotly.graph_objs as go
from morpheus.algorithms.logistic_regression import NormalizedLogisticRegression
from morpheus.algorithms.linear_regression import NormalizedLinearRegression
from morpheus.algorithms.kmeans import NormalizedKMeans
from morpheus.algorithms.GNMF import GaussianNMF

trails = 10
nr = 1000000
ds = 20

# Linear regression
print "start testing linear regression on synthetic data"
total = []
for f in range(1, 5):
    result = []
    for t in range(1, 21):
        print "Linear regression, feature ratio:", f, "tuple ratio", t
        dr = ds * f
        ns = nr * t

        s = np.random.rand(ns, ds)
        r = [np.random.rand(nr, dr)]
        num = np.random.randint(nr, size=ns)
        while (max(num) != nr - 1):
            num = np.random.randint(nr, size=ns)
        k = [num]
        T = np.mat(np.hstack((s, r[0][k[0]])))
        Y = np.matrix(np.random.randint(2, size=ns)).T
        normalized_matrix = nm.NormalizedMatrix(s, r, k)
        avg = []
        for _ in range(trails):
            w_init_m = np.matrix(np.random.randn(T.shape[1], 1))
            m_regressor = NormalizedLinearRegression()
            m_start = time.time()
            m_regressor.fit(T, Y, w_init=w_init_m)
            m_end = time.time()

            w_init_n = np.matrix(np.random.randn(T.shape[1], 1))
            n_regressor = NormalizedLinearRegression()
            n_start = time.time()
            n_regressor.fit(normalized_matrix, Y, w_init=w_init_n)
            n_end = time.time()
            avg.append((m_end - m_start) / (n_end - n_start))

        print (sum(avg) - min(avg) - max(avg)) / (trails - 2)
        result.append((sum(avg) - min(avg) - max(avg)) / (trails - 2))
    total.append(result)

layout = go.Layout(
    title='LS speedup on synthetic data',
    xaxis=dict(
        title='Tuple Ratio'
    ),
    yaxis=dict(
        title='Feature Ratio'
    )
)
trace = go.Heatmap(z=total,
                   x=range(1, 21),
                   y=range(1, 5),
                   zmin=0,
                   zmax=3)
data = [trace]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='ls_synthetic.html', show_link=False)

# Logistic regression
print "start testing synthetic data on logistic regression"
total = []
for f in range(1, 5):
    result = []
    for t in range(1, 21):
        print "Logistic regression, feature ratio:", f, "tuple ratio", t
        dr = ds * f
        ns = nr * t

        s = np.random.rand(ns, ds)
        r = [np.random.rand(nr, dr)]
        num = np.random.randint(nr, size=ns)
        while (max(num) != nr - 1):
            num = np.random.randint(nr, size=ns)
        k = [num]
        T = np.mat(np.hstack((s, r[0][k[0]])))
        Y = np.matrix(np.random.randint(2, size=ns)).T
        normalized_matrix = nm.NormalizedMatrix(s, r, k)
        avg = []
        for _ in range(trails):
            w_init_m = np.matrix(np.random.randn(T.shape[1], 1))
            m_regressor = NormalizedLogisticRegression()
            m_start = time.time()
            m_regressor.fit(T, Y, w_init=w_init_m)
            m_end = time.time()

            w_init_n = np.matrix(np.random.randn(T.shape[1], 1))
            n_regressor = NormalizedLogisticRegression()
            n_start = time.time()
            n_regressor.fit(normalized_matrix, Y, w_init=w_init_n)
            n_end = time.time()
            avg.append((m_end - m_start) / (n_end - n_start))

        print (sum(avg) - min(avg) - max(avg)) / (trails - 2)
        result.append((sum(avg) - min(avg) - max(avg)) / (trails - 2))
    total.append(result)

layout = go.Layout(
    title='LR speedup on synthetic data',
    xaxis=dict(
        title='Tuple Ratio'
    ),
    yaxis=dict(
        title='Feature Ratio'
    )
)
trace = go.Heatmap(z=total,
                   x=range(1, 21),
                   y=range(1, 5),
                   zmin=0,
                   zmax=3)
data = [trace]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='lr_synthetic.html', show_link=False)


# K Means
print "start testing on K means on synthetic data"
total = []
for f in range(1, 5):
    result = []
    for t in range(1, 21):
        print "Logistic regression, feature ratio:", f, "tuple ratio", t
        dr = ds * f
        ns = nr * t

        s = np.random.rand(ns, ds)
        r = [np.random.rand(nr, dr)]
        num = np.random.randint(nr, size=ns)
        while (max(num) != nr - 1):
            num = np.random.randint(nr, size=ns)
        k = [num]
        T = np.mat(np.hstack((s, r[0][k[0]])))
        Y = np.matrix(np.random.randint(2, size=ns)).T
        normalized_matrix = nm.NormalizedMatrix(s, r, k)
        avg = []
        for _ in range(trails):
            center_number = 5
            k_center_m = (T[:center_number, : ns + max(k[0])]).T
            m_regressor = NormalizedKMeans()
            m_start = time.time()
            m_regressor.fit(T, k_center_m)
            m_end = time.time()

            k_center_n = (T[:center_number, : ns + max(k[0])]).T
            n_regressor = NormalizedKMeans()
            n_start = time.time()
            n_regressor.fit(normalized_matrix, k_center_n)
            n_end = time.time()
            avg.append((m_end - m_start) / (n_end - n_start))

        print (sum(avg) - min(avg) - max(avg)) / (trails - 2)
        result.append((sum(avg) - min(avg) - max(avg)) / (trails - 2))
    total.append(result)

layout = go.Layout(
    title='Kmeans speedup on synthetic data',
    xaxis=dict(
        title='Tuple Ratio'
    ),
    yaxis=dict(
        title='Feature Ratio'
    )
)
trace = go.Heatmap(z=total,
                   x=range(1, 21),
                   y=range(1, 5),
                   zmin=0,
                   zmax=3)
data = [trace]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='kmeans_synthetic.html', show_link=False)


# GNMF
print "start testing GNMF on synthetic data"
total = []
for f in range(1, 5):
    result = []
    for t in range(1, 21):
        print "GNMF, feature ratio:", f, "tuple ratio", t
        dr = ds * f
        ns = nr * t

        s = np.random.rand(ns, ds)
        r = [np.random.rand(nr, dr)]
        num = np.random.randint(nr, size=ns)
        while (max(num) != nr - 1):
            num = np.random.randint(nr, size=ns)
        k = [num]
        T = np.mat(np.hstack((s, r[0][k[0]])))
        Y = np.matrix(np.random.randint(2, size=ns)).T
        normalized_matrix = nm.NormalizedMatrix(s, r, k)
        avg = []
        for _ in range(trails):
            w_init_m = np.matrix(np.random.randn(T.shape[0], 5))
            h_init_m = np.mat(np.random.rand(5, T.shape[1]))
            m_regressor = GaussianNMF()
            m_start = time.time()
            m_regressor.fit(T, w_init=w_init_m, h_init=h_init_m)
            m_end = time.time()

            w_init_n = np.matrix(np.random.randn(T.shape[0], 5))
            h_init_n = np.mat(np.random.rand(5, T.shape[1]))
            n_regressor = GaussianNMF()
            n_start = time.time()
            n_regressor.fit(normalized_matrix, w_init=w_init_n, h_init=h_init_n)
            n_end = time.time()
            avg.append((m_end - m_start) / (n_end - n_start))

        print (sum(avg) - min(avg) - max(avg)) / (trails - 2)
        result.append((sum(avg) - min(avg) - max(avg)) / (trails - 2))
    total.append(result)

layout = go.Layout(
    title='GNMF speedup on synthetic data',
    xaxis=dict(
        title='Tuple Ratio'
    ),
    yaxis=dict(
        title='Feature Ratio'
    )
)
trace = go.Heatmap(z=total,
                   x=range(1, 21),
                   y=range(1, 5),
                   zmin=0,
                   zmax=3)
data = [trace]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='gnmf_synthetic.html', show_link=False)