#pragma once

#include <cassert>
#include <Python.h>

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL FIXPRS_ARRAY_API
#include <numpy/arrayobject.h>

#include "column.hh"

//------------------------------------------------------------------------------

struct Result
{
  // Number of error values.
  size_t num_err = 0;
  // Location and value of the first error value.
  size_t err_idx = -1;
  std::string err_val;
};

class Array
{
public:

  Array(
    size_t const width,
    size_t const len)
  : width_(width)
  , idx_(0)
  {
    npy_intp l = len;
    arr_ = PyArray_New(
      &PyArray_Type, 1, &l, NPY_STRING, nullptr, nullptr, width, 0, nullptr);
    assert(arr_ != nullptr);  // FIXME
    ptr_ = (char*) PyArray_DATA((PyArrayObject*) arr_);
    stride_ = width;
  }

  ~Array() {
    ptr_ = nullptr;
    Py_XDECREF(arr_);
    arr_ = nullptr;
  }

  Array(Array const&) = delete;
  Array(Array&&) = default;
  void operator=(Array const&) = delete;
  Array& operator=(Array&&) = default;

  void expand(size_t const len) {
    assert(arr_ != nullptr);
    if (len > (size_t) PyArray_SIZE((PyArrayObject*) arr_))
      abort();
  }

  PyObject* release(size_t const len) {
    assert(arr_ != nullptr);
    // FIXME: Resize.
    auto const arr = arr_;
    arr_ = nullptr;
    ptr_ = nullptr;
    return arr;
  }

  Result parse(Column const& col) {
    for (auto field : col) {
      auto const ptr = ptr_ + idx_ * stride_;
      // Copy the bytes in.
      memcpy(ptr, field.ptr, field.len);
      // Zero out the rest of the field.
      memset(ptr + field.len, 0, width_ - field.len);
      // Advance.
      ++idx_;
    }

    return Result{};
  }

private:

  size_t width_;
  size_t idx_;

  PyObject* arr_;
  char* ptr_;
  size_t stride_;

};


extern Result parse_bytes(Column const& col, Array& arr);

