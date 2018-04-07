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

trails = 10
nr = 1000000
ds = 20

# Scalar
print "start tesing scalar"
total = []
for f in range(1, 5):
    result = []
    for t in range(1, 21):
        print "scalar, feature ratio:", f, "tuple ratio", t
        dr = ds * f
        ns = nr * t

        s = np.random.rand(ns, ds)
        r = [np.random.rand(nr, dr)]
        num = np.random.randint(nr, size=ns)
        k = [num]
        T = np.hstack((s, r[0][k[0]]))
        normalized_matrix = nm.NormalizedMatrix(s, r, k)

        avg = []
        for _ in range(trails):
            m_start = time.time()
            # np.add(T, 5)
            np.power(T, 2)
            m_end = time.time()

            n_start = time.time()
            # normalized_matrix + 5
            np.power(normalized_matrix, 2)
            n_end = time.time()

            avg.append((m_end - m_start) / (n_end - n_start))

            print (m_end - m_start) / (n_end - n_start)
        result.append((sum(avg) - min(avg) - max(avg)) / (trails - 2))
    total.append(result)

layout = go.Layout(
    title='Scalar speedup',
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
                   zmax=6)

data = [trace]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='scalar.html', show_link=False)


print "start tesing LMM"
total = []
for f in range(1, 5):
    result = []
    for t in range(1, 21):
        print "LMM, feature ratio:", f, "tuple ratio", t
        dr = ds * f
        ns = nr * t
        # s = np.matrix([])
        s = np.random.rand(ns, ds)
        r = [np.random.rand(nr, dr)]

        # s = sparse.rand(ns, ds, 0.0001, format="cso")
        # r = [sparse.rand(nr, dr, 0.0001, format="cso")]
        # print s.shape, r.shape
        num = np.random.randint(nr, size=ns)
        while (max(num) != nr - 1):
            num = np.random.randint(nr, size=ns)
        k = [num]
        # T = sparse.hstack((s, r[0][k[0][0]])).tocsr()
        T = np.hstack((s, r[0][k[0]]))
        normalized_matrix = nm.NormalizedMatrix(s, r, k)
        w_init = np.matrix(np.random.randn(T.shape[1], 1))

        avg = []
        for _ in range(trails):
            m_start = time.time()
            T.dot(w_init)
            m_end = time.time()

            n_start = time.time()
            normalized_matrix.dot(w_init)
            n_end = time.time()
            avg.append((m_end - m_start) / (n_end - n_start))
        result.append((sum(avg) - min(avg) - max(avg)) / (trails - 2))
    total.append(result)

layout = go.Layout(
    title='LMM speedup',
    xaxis=dict(
        title='Feature Ratio'
    ),
    yaxis=dict(
        title='Tuple Ratio'
    )
)
trace = go.Heatmap(z=total,
                   x=range(1, 21),
                   y=range(1, 5),
                   zmin=0,
                   zmax=6)
data = [trace]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='lmm.html', show_link=False)

# RMM
print "start tesing RMM"
total = []
for f in range(1, 5):
    result = []
    for t in range(1, 21):
        print "RMM, feature ratio:", f, "tuple ratio", t
        dr = ds * f
        ns = nr * t

        s = np.random.rand(ns, ds)
        r = [np.random.rand(nr, dr)]
        num = np.random.randint(nr, size=ns)
        k = [num]
        T = np.hstack((s, r[0][k[0]]))
        normalized_matrix = nm.NormalizedMatrix(s, r, k)


        avg = []
        for _ in range(trails):
            w_init1 = np.matrix(np.random.randn(1, ns))
            m_start = time.time()
            np.dot(w_init1, T)
            m_end = time.time()

            w_init2 = np.matrix(np.random.randn(1, ns))
            n_start = time.time()
            w_init2 * normalized_matrix
            n_end = time.time()
            avg.append((m_end - m_start) / (n_end - n_start))
        result.append((sum(avg) - min(avg) - max(avg)) / (trails - 2))
    total.append(result)
layout = go.Layout(
    title='RMM speedup',
    xaxis=dict(
        title='Feature Ratio'
    ),
    yaxis=dict(
        title='Tuple Ratio'
    )
)

trace = go.Heatmap(z=total,
                   x=range(1, 21),
                   y=range(1, 5),
                   zmin=0,
                   zmax=6)
data = [trace]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='rmm.html', show_link=False)

# Cross product
print "start tesing cross product (m.T * m)"
total = []
for f in range(1, 5):
    result = []
    for t in range(1, 21):
        print "Crossprod, feature ratio:", f, "tuple ratio", t
        dr = ds * f
        ns = nr * t

        s = np.random.rand(ns, ds)
        r = [np.random.rand(nr, dr)]
        num = np.random.randint(nr, size=ns)
        while (max(num) != nr - 1):
            num = np.random.randint(nr, size=ns)
        k = [num]
        T = np.hstack((s, r[0][k[0]]))
        normalized_matrix = nm.NormalizedMatrix(s, r, k)
        # print num
        avg = []
        for _ in range(trails):
            m_start = time.time()
            np.dot(T.T, T)
            # tmp0 = T.T * T
            m_end = time.time()
            n_start = time.time()
            tmp1 = normalized_matrix.T * normalized_matrix
            n_end = time.time()
            avg.append((m_end - m_start) / (n_end - n_start))

        print (sum(avg) - min(avg) - max(avg)) / (trails - 2)
        result.append((sum(avg) - min(avg) - max(avg)) / (trails - 2))
    total.append(result)

layout = go.Layout(
    title='Crossprod speedup',
    xaxis=dict(
        title='Feature Ratio'
    ),
    yaxis=dict(
        title='Tuple Ratio'
    )
)
trace = go.Heatmap(z=total,
                   x=range(1, 21),
                   y=range(1, 5),
                   zmin=0,
                   zmax=6)
data = [trace]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='crossprod.html', show_link=False)


# Inverse
print "start tesing inverse"
total = []
for f in range(1, 5):
    result = []
    for t in range(1, 21):
        print "inverse, feature ratio:", f, "tuple ratio", t
        dr = ds * f
        ns = nr * t

        s = np.random.rand(ns, ds)
        r = [np.random.rand(nr, dr)]
        num = np.random.randint(nr, size=ns)
        while (max(num) != nr - 1):
            num = np.random.randint(nr, size=ns)
        k = [num]
        T = np.mat(np.hstack((s, r[0][k[0]])))
        normalized_matrix = nm.NormalizedMatrix(s, r, k)
        # print num
        avg = []
        for _ in range(trails):
            m_start = time.time()
            tmp = T.I
            m_end = time.time()
            n_start = time.time()
            tmp2 = normalized_matrix.I
            n_end = time.time()
            avg.append((m_end - m_start) / (n_end - n_start))

        print (sum(avg) - min(avg) - max(avg)) / (trails - 2)
        result.append((sum(avg) - min(avg) - max(avg)) / (trails - 2))
    total.append(result)

layout = go.Layout(
    title='Inverse speedup',
    xaxis=dict(
        title='Feature Ratio'
    ),
    yaxis=dict(
        title='Tuple Ratio'
    )
)
trace = go.Heatmap(z=total,
                   x=range(1, 21),
                   y=range(1, 5),
                   zmin=0,
                   zmax=6)
data = [trace]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='inverse.html', show_link=False)
