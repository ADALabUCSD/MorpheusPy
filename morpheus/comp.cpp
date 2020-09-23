// Copyright 2018 Side Li and Arun Kumar
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Python.h>
#include <vector>
#include <numeric>
#include <iterator>
#include <iostream>
#include <math.h>
#include "numpy/arrayobject.h"

extern "C" {
    void initcomp(void);
}

template <class I, class T, class K>
static void add_new(const I ns,
                    const I nk,
                    const I dw,
                    const std::vector<K*> k,
                    const std::vector<T*> v,
                    const std::vector<K> vd,
                          T res[])
{
    for (int j = 0; j < nk; j++) {
        for (int c = 0; c < dw; c++) {
            long o1 = c * ns;
            long o2 = c * vd[j];
            for (int i = 0; i < ns; i++) {
                res[o1++] += v[j][k[j][i] + o2];
            }
        }
    }
//  Another implementation that exploits other memory layout
//    long o = 0;
//    for (int i = 0; i < ns; i++) {
//        for (int j = 0; j < dw; j++) {
//            for (int m = 0; m < nk; m++) {
//                res[o] += v[m][k[m][i] * vd[m] + j];
//            }
//            o++;
//        }
//    }
}

template <class I, class T, class K>
static void expand_add(const I ns,
                    const I nk,
                    const std::vector<K*> k,
                    const std::vector<T*> r,
                    const std::vector<K> nr,
                          T res[])
{
    long o = 0;
    for (int i = 0; i < ns; i++) {
        for (int j = 0; j < ns; j++) {
            for (int m = 0; m < nk; m++) {
                res[o++] += r[m][nr[m] * k[m][i] + k[m][j]];
            }
        }
    }
}

template <class I, class T, class K>
static void group_left(const I ns,
                    const I ds,
                    const T* s,
                    const K* k,
                          T res[])
{
    long o1 = 0;
    for (int i = 0; i < ns; i++) {
        long o2 = k[i] * ds;
        for (int j = 0; j < ds; j++) {
            res[o2++] += s[o1++];
        }
    }
}

template <class I, class T, class K>
static void group_k_by_k(const I n,
                    const I d,
                    const I ns,
                    const K* ki,
                    const K* kj,
                          T res[])
{
    for(int i = 0; i < ns; i++) {
        res[kj[i] + ki[i] * d] += 1;
    }
}

template <class I, class T, class K>
static void group_k_by_k_w(const I n,
                    const I d,
                    const I ns,
                    const T* w,
                    const K* ki,
                    const K* kj,
                          T res[])
{
    for(int i = 0; i < ns; i++) {
        res[kj[i] + ki[i] * d] += w[i];
    }
}

template <class I, class T, class K>
static void group(const I ns,
                  const I nk,
                  const I nw,
                  const std::vector<K*> k,
                  const std::vector<I> nr,
                  const T* w,
                        std::vector<T*> res)
{
    for (int j = 0; j < nk; j++) {
        long o1 = 0;
        long o2 = 0;
        for (int r = 0; r < nw; r++) {
            for (int i = 0; i < ns; i++) {
                res[j][o1 + k[j][i]] += w[o2++];
            }
            o1 += nr[j];
        }
    }
}

template <class I, class T>
static void multiply(const I nr,
                     const I dr,
                     const T* r,
                     const T* v,
                     T* res)
{
    int o = 0;
    for (int i = 0; i < nr; i++) {
        T scalar = sqrt(v[i]);
        for (int j = 0; j < dr; j++) {
            res[o] = r[o] * scalar;
            o++;
        }
    }
}

template <class I, class T>
static void multiply_sparse(const I n,
                     const I* rows,
                     const T* data,
                     const T* v,
                     T* res)
{
    for (int i = 0; i < n; i++) {
        res[i] = data[i] * v[rows[i]];
    }
}

static PyObject * add_new(PyObject *self, PyObject* args)
{
    int ns;
    int nk;
    int dw;
    PyObject* k;
    PyObject* vd;
    PyObject* v;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiiOOOO", &ns, &nk, &dw, &k, &v, &vd, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    std::vector<long*> k_list;
    std::vector<double*> v_list;
    std::vector<long> vd_list;
    for(int i = 0; i < nk; i++) {
          k_list.push_back((long*) PyArray_DATA(PyList_GET_ITEM(k, i)));
          v_list.push_back((double*) PyArray_DATA(PyList_GET_ITEM(v, i)));
          vd_list.push_back(PyInt_AsLong(PyList_GET_ITEM(vd, i)));
    }

    add_new<int, double, long>(ns, nk, dw, k_list, v_list, vd_list, (double*)PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * group_left(PyObject *self, PyObject* args)
{
    int ns;
    int ds;
    PyObject* k;
    PyObject* s;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiOOO", &ns, &ds, &s, &k, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    group_left<int, double, long>(ns, ds, (double*) PyArray_DATA(s), (long*) PyArray_DATA(k), (double*)PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * group_k_by_k(PyObject *self, PyObject* args)
{
    int n;
    int d;
    int ns;
    PyObject* ki;
    PyObject* kj;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiiOOO", &n, &d, &ns, &ki, &kj, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    group_k_by_k<int, double, long>(n, d, ns, (long*) PyArray_DATA(ki), (long*) PyArray_DATA(kj), (double*)PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * group_k_by_k_w(PyObject *self, PyObject* args)
{
    int n;
    int d;
    int ns;
    PyObject* w;
    PyObject* ki;
    PyObject* kj;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiiOOOO", &n, &d, &ns, &w, &ki, &kj, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    group_k_by_k_w<int, double, long>(n, d, ns, (double*) PyArray_DATA(w), (long*) PyArray_DATA(ki), (long*) PyArray_DATA(kj), (double*)PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * expand_add(PyObject *self, PyObject* args)
{
    int ns;
    int nk;
    PyObject* k;
    PyObject* r;
    PyObject* nr;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiOOOO", &ns, &nk, &k, &r, &nr, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    std::vector<long*> k_list;
    std::vector<double*> r_list;
    std::vector<long> nr_list;
    for(int i = 0; i < nk; i++) {
          k_list.push_back((long*) PyArray_DATA(PyList_GET_ITEM(k, i)));
          r_list.push_back((double*) PyArray_DATA(PyList_GET_ITEM(r, i)));
          nr_list.push_back(PyInt_AsLong(PyList_GET_ITEM(nr, i)));
    }

    expand_add<int, double, long>(ns, nk, k_list, r_list, nr_list, (double*)PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * group(PyObject *self, PyObject* args)
{
    int ns;
    int nk;
    int nw;
    PyObject* k;
    PyObject* w;
    PyObject* res;
    PyObject* nr;

    if (!PyArg_ParseTuple(args, "iiiOOOO", &ns, &nk, &nw, &k, &nr, &w, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    std::vector<long*> k_list;
    std::vector<double*> res_list;
    std::vector<int> nr_list;
    for(int i = 0; i < nk; i++) {
          k_list.push_back((long*) PyArray_DATA(PyList_GET_ITEM(k, i)));
          res_list.push_back((double*) PyArray_DATA(PyList_GET_ITEM(res, i)));
          nr_list.push_back((long) PyInt_AsLong(PyList_GET_ITEM(nr, i)));
    }

    group<int, double, long>(ns, nk, nw, k_list, nr_list, (double*) PyArray_DATA(w), res_list);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * multiply(PyObject *self, PyObject* args)
{
    int nr;
    int dr;
    PyObject* r;
    PyObject* v;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiOOO", &nr, &dr, &r, &v, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    multiply<int, double>(nr, dr, (double*) PyArray_DATA(r), (double *) PyArray_DATA(v), (double *) PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * multiply_sparse(PyObject *self, PyObject* args)
{
    int n;
    PyObject* rows;
    PyObject* data;
    PyObject* v;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iOOOO", &n, &rows, &data, &v, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    multiply_sparse<int, double>(n,
                          (int*) PyArray_DATA(rows), (double*) PyArray_DATA(data),
                          (double *) PyArray_DATA(v), (double *) PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef comp_methods[] = {
	{"add_new", add_new,    METH_VARARGS,
	 "add v list to res"},
	{"expand_add", expand_add,    METH_VARARGS,
	 "expand nr*ns matrix to ns*ns"},
	{"group_left", group_left, METH_VARARGS,
	 "group ns*ds matrix to nr*ds"},
	{"group_k_by_k", group_k_by_k, METH_VARARGS,
	 "group ki by kj"}, {"group_k_by_k_w", group_k_by_k_w, METH_VARARGS, "group ki by kj with Weight"},
	{"group", group, METH_VARARGS,
	 "group in rmm"},
	{"multiply", multiply, METH_VARARGS,
	 "multiply dense matrix with scalar vector"},
	{"multiply_sparse", multiply_sparse, METH_VARARGS,
	 "multiply coo sparse matrix with scalar vector"},
	{NULL,		NULL}		/* sentinel */
};

extern void initcomp(void)
{
	PyImport_AddModule("comp");
	Py_InitModule("comp", comp_methods);
}

int main(int argc, char **argv)
{
	Py_SetProgramName(argv[0]);

	Py_Initialize();

	initcomp();

	Py_Exit(0);
}