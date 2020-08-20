#include <cassert>
#include <Python.h>

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL FIXPRS_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

#include "array.hh"

//------------------------------------------------------------------------------

BytesArray::BytesArray(
  size_t const len,
  size_t const width,
  Config const& cfg)
: Array(len),
  width_(width),
  resize_cfg_(cfg.resize)
{
  npy_intp l = len;
  arr_ = PyArray_New(
    &PyArray_Type, 1, &l, NPY_STRING, nullptr, nullptr, width, 0, nullptr);
  assert(arr_ != nullptr);  // FIXME
  ptr_ = (char*) PyArray_DATA((PyArrayObject*) arr_);
  stride_ = width;
}


BytesArray::~BytesArray() {
  ptr_ = nullptr;
  Py_XDECREF(arr_);
  arr_ = nullptr;
}


void
BytesArray::resize(
  size_t const len)
{
  assert(arr_ != nullptr);

  npy_intp shape[1] = {(npy_intp) len};
  PyArray_Dims dims = { shape, 1 };
  // FIXME: Not sure what the return value of PyArray_Resize is.  This function
  // resizes the array in place.  
  PyArray_Resize((PyArrayObject*) arr_, &dims, 0, NPY_CORDER);

  len_ = len;
  // Possibly new pointer to data.
  ptr_ = (char*) PyArray_DATA((PyArrayObject*) arr_);
}


PyObject*
BytesArray::release()
{
  auto const arr = arr_;

  assert(arr != nullptr);
  if (idx_ < len_) {
    // Trim the array to length.
    // FIXME: Slice rather than resize in some cases?
    npy_intp shape[1] = {(npy_intp) idx_};
    PyArray_Dims dims = { shape, 1 };
    PyArray_Resize((PyArrayObject*) arr, &dims, 0, NPY_CORDER);
  }

  arr_ = nullptr;
  ptr_ = nullptr;
  return arr;
}


