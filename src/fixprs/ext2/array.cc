#include <cassert>
#include <Python.h>

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL FIXPRS_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

#include "array.hh"

//------------------------------------------------------------------------------

ArraysTarget::~ArraysTarget()
{
  for (auto&& arr : arrs_)
    Py_XDECREF(arr);
}


void
ArraysTarget::add_col(
  int const type_num,
  int const itemsize)
{
  npy_intp shape[1] = {(npy_intp) len_};
  auto arr = PyArray_New(
    &PyArray_Type, 1, shape, type_num, nullptr, nullptr, itemsize, 0, nullptr);
  assert(arr != nullptr);  // FIXME

  arrs_.push_back(arr);
}


void
ArraysTarget::resize(
  size_t const len)
{
  for (auto arr : arrs_) {
    npy_intp shape[1] = {(npy_intp) len};
    PyArray_Dims dims = {shape, 1};
    // FIXME: Not sure what the return value of PyArray_Resize is.  This
    // function resizes the array in place.
    PyArray_Resize((PyArrayObject*) arr, &dims, 0, NPY_CORDER);
  }

  len_ = len;
}


ArraysTarget::Ptr
ArraysTarget::get_pointer(
  size_t const c)
const
{
  auto const arr = (PyArrayObject*) arrs_[c];
  return {(char*) PyArray_DATA(arr), PyArray_STRIDES(arr)[0]};
}


PyObject*
ArraysTarget::release()
{
  // Extract and package up arrays.
  auto arrs = PyTuple_New(arrs_.size());
  if (arrs == nullptr)
    return nullptr;

  size_t i = 0;
  for (auto&& arr : arrs_)
    PyTuple_SET_ITEM(arrs, i++, arr);
  arrs_.clear();

  return arrs;
}


