#pragma once

#include <cassert>
#include <Python.h>

#include "column.hh"
#include "parse_int.hh"

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
    size_t const len,
    Config const& cfg)
  : resize_cfg_(cfg.resize),
    len_(len),
    idx_(0)
  {
  }

  virtual ~Array() {}

  size_t size() const { return idx_; }

  virtual void resize(size_t len) = 0;
  virtual Result parse(Column const& col) = 0;
  virtual PyObject* release() = 0;

  virtual void expand(
    size_t const len)
  {
    if (resize_cfg_.grow) {
      auto l = len_;
      while (l < len)
        l = std::max(
          (size_t) (l * resize_cfg_.grow_factor),
          l + resize_cfg_.min_grow);
      resize(l);
    }
    else
      // FIXME
      abort();
  }


  bool check_size(
    size_t const len)
  {
    if (unlikely(len_ < len)) {
      expand(len);
      return true;
    }
    else
      return false;
  }


protected:

  ArrayResizeConfig resize_cfg_;

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


  virtual void resize(size_t len);
  virtual PyObject* release();

private:

  size_t width_;

  PyObject* arr_;
  char* ptr_;
  size_t stride_;

};


//------------------------------------------------------------------------------

/*
 * Array for dtype kind 'i': int64.
 */
class Int64Array
: public Array
{
public:

  Int64Array(size_t const len, Config const& cfg);
  virtual ~Int64Array();

  Int64Array(Int64Array const&) = delete;
  void operator=(Int64Array const&) = delete;

  Int64Array(Int64Array&&) = delete;
  void operator=(Int64Array&&) = delete;

  virtual Result parse(
    Column const& col)
  {
    Result res;

    for (auto field : col) {
      auto const ptr = (long*) (ptr_ + idx_ * stride_);
      auto const val = ::parse<int64_t>(field);
      if (likely(val))
        *ptr = *val;
      else if (res.num_err++ == 0) {
        res.first_err_idx = idx_;
        res.first_err_val = std::string(field.ptr, field.len);
      }

      ++idx_;
    }

    return res;
  }


  virtual void resize(size_t len);
  virtual PyObject* release();

private:

  PyObject* arr_;
  char* ptr_;
  size_t stride_;

};

