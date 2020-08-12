#include <cassert>
#include <Python.h>

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL FIXPRS_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

#include "array.hh"

//------------------------------------------------------------------------------

Array::Array(
  size_t const width,
  size_t const len)
: width_(width)
, len_(len)
, idx_(0)
{
  npy_intp l = len;
  arr_ = PyArray_New(
    &PyArray_Type, 1, &l, NPY_STRING, nullptr, nullptr, width, 0, nullptr);
  assert(arr_ != nullptr);  // FIXME
  ptr_ = (char*) PyArray_DATA((PyArrayObject*) arr_);
  stride_ = width;
}


Array::~Array() {
  ptr_ = nullptr;
  Py_XDECREF(arr_);
  arr_ = nullptr;
}


PyObject*
Array::release(
  size_t const len)
{
  assert(arr_ != nullptr);
  // FIXME: Resize.
  auto const arr = arr_;
  arr_ = nullptr;
  ptr_ = nullptr;
  return arr;
}


