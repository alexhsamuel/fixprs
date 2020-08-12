#pragma once

#include <cassert>
#include <Python.h>

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

  Array(size_t const width, size_t const len);
  ~Array();

  Array(Array const&) = delete;
  Array(Array&&) = default;
  void operator=(Array const&) = delete;
  Array& operator=(Array&&) = default;

  void expand(size_t const len) {
    assert(arr_ != nullptr);
    if (len > len_)
      // FIXME: Expand.
      abort();
  }

  PyObject* release(size_t const len);

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
  size_t len_;
  size_t idx_;

  PyObject* arr_;
  char* ptr_;
  size_t stride_;

};


extern Result parse_bytes(Column const& col, Array& arr);

