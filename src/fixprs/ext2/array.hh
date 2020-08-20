#pragma once

#include <cassert>
#include <Python.h>

#include "column.hh"

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

//------------------------------------------------------------------------------

struct Result
{
  // Number of error values.
  size_t num_err = 0;
  // Location and value of the first error value.
  size_t first_err_idx = -1;
  std::string first_err_val;
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

  virtual bool expand(size_t len) = 0;
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

  BytesArray(size_t const len, size_t const width, Config const& cfg);
  virtual ~BytesArray();

  BytesArray(BytesArray const&) = delete;
  void operator=(BytesArray const&) = delete;

  BytesArray(BytesArray&&) = delete;
  BytesArray& operator=(BytesArray&&) = delete;

  virtual bool expand(
    size_t const len)
  {
    if (len_ < len) {
      auto l = std::max(len_, 1ul);
      // FIXME: Tune.
      while (l < len)
        l *= 2;
      resize(l);
      return true;
    }
    else
      return false;
  }


  virtual Result parse(
    Column const& col)
  {
    Result res;

    for (auto field : col) {
      auto const len = std::min(field.len, width_);
      auto const ptr = ptr_ + idx_ * stride_;
      // Copy the bytes in.
      memcpy(ptr, field.ptr, len);
      // Zero out the rest of the field.
      memset(ptr + len, 0, width_ - len);

      if (unlikely(field.len > width_)) {
        if (res.num_err++ == 0) {
          res.first_err_idx = idx_;
          res.first_err_val = std::string(field.ptr, field.len);
        }
      }

      // Advance.
      ++idx_;
    }

    return res;
  }


  virtual PyObject* release();

private:

  void resize(size_t len);

  size_t width_;
  ArrayResizeConfig resize_cfg_;

  PyObject* arr_;
  char* ptr_;
  size_t stride_;

};


