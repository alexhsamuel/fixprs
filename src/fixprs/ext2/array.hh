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
    size_t const initial_len)
  : len_(initial_len)
  {
  }

  virtual ~Target() {};

  size_t length() const { return len_; }
  virtual size_t num_cols() const = 0;

  /* Adds a new col array with `typenum`.  */
  virtual void add_col(int typenum, int itemsize=0) = 0;

  /* Resizes the array to `len`.  */
  virtual void resize(size_t len) = 0;

  /* Returns the pointer and stride for contents.  */
  virtual Ptr get_pointer(size_t c) const = 0;

  /* Releases the underlying array and invalidates this.  */
  virtual PyObject* release() = 0;

protected:

  /* Current array length.  */
  size_t len_;

};


//------------------------------------------------------------------------------

class ArraysTarget
: public Target
{
public:

  ArraysTarget(
    size_t const initial_len)
  : Target(initial_len)
  {
  };

  virtual ~ArraysTarget();
  virtual size_t num_cols() const { return arrs_.size(); }
  virtual void add_col(int typenum, int itemsize=0);
  virtual void resize(size_t len);
  virtual Ptr get_pointer(size_t col) const;
  virtual PyObject* release();

private:

  static PyObject* make_col(size_t col, size_t initial_len);

  std::vector<PyObject*> arrs_;

};


