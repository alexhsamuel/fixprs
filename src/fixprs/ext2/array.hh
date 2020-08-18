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

//------------------------------------------------------------------------------

class Array
{
public:

  Array(
    size_t const len)
  : len_(len),
    idx_(0)
  {}

  virtual ~Array() {}

  size_t size() const { return idx_; }

  virtual void expand(size_t len) = 0;
  virtual Result parse(Column const& col) = 0;
  virtual PyObject* release() = 0;

protected:

  size_t len_;
  size_t idx_;

};

//------------------------------------------------------------------------------

/*
 * Array for dtype kind 'S': bytes.
 */
class BytesArray
: public Array
{
public:

  BytesArray(size_t const len, size_t const width);
  virtual ~BytesArray();

  BytesArray(BytesArray const&) = delete;
  void operator=(BytesArray const&) = delete;

  BytesArray(BytesArray&&) = delete;
  BytesArray& operator=(BytesArray&&) = delete;

  virtual void expand(
    size_t const len)
  {
    if (len_ < len) {
      auto l = std::max(len_, 1ul);
      while (l < len)
        l *= 2;
      // FIXME: Tune.
      resize(l);
    }
  }


  virtual Result parse(
    Column const& col)
  {
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


  virtual PyObject* release();

private:

  void resize(size_t len);

  size_t width_;

  PyObject* arr_;
  char* ptr_;
  size_t stride_;

};


