#pragma once

#include <cassert>
#include <Python.h>

#include "column.hh"
#include "parse_int.hh"

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

//------------------------------------------------------------------------------

class Target
{
public:

  using Ptr = std::pair<char*, size_t>;

  Target(
    size_t const initial_len,
    Config const& cfg)
  : len_(initial_len),
    resize_cfg_(cfg.resize)
  {
  }

  virtual ~Target() {};

  virtual size_t num_cols() const = 0;

  /* Adds a new col array with `typenum`.  */
  virtual void add_col(int typenum, int itemsize=0) = 0;

  /* Resizes the array to `len`.  */
  virtual void resize(size_t len) = 0;

  /* Returns the pointer and stride for contents.  */
  virtual Ptr get_pointer(size_t c) const = 0;

  /* Releases the underlying array and invalidates this.  */
  virtual PyObject* release(size_t len) = 0;

  // FIXME: Move out, along with `expand` and `resize_cfg_`.
  /*
   * Confirms that the array accommodates `len`; if not, expands it.
   */
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

  /*
   * Expands the array to at least `len`.
   */
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

  /* Current array length.  */
  size_t len_;

private:

  ResizeConfig const resize_cfg_;

};


//------------------------------------------------------------------------------

class ArraysTarget
: public Target
{
public:

  ArraysTarget(
    size_t const initial_len,
    Config const& cfg)
  : Target(initial_len, cfg)
  {
  };

  virtual ~ArraysTarget();
  virtual size_t num_cols() const { return arrs_.size(); }
  virtual void add_col(int typenum, int itemsize=0);
  virtual void resize(size_t len);
  virtual Ptr get_pointer(size_t col) const;
  virtual PyObject* release(size_t len);

private:

  static PyObject* make_col(size_t col, size_t initial_len);

  std::vector<PyObject*> arrs_;

};


