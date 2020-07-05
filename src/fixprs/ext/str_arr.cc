#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <Python.h>
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL FIXPRS_ARRAY_API
#include <numpy/arrayobject.h>

#include "str_arr.hh"

//------------------------------------------------------------------------------

std::pair<std::string, PyObject*>
parse_str_col(
  Column const& col)
{
  npy_intp len = col.size();

  auto const header = true;  // FIXME: Configuration.

  size_t i = 0;
  auto fields = col.begin();

  std::string name = "???";  // FIXME
  if (header) {
    if (len == 0)
      // Missing header.
      // FIXME: Check this earlier?
      return {};

    auto const& n = *fields;
    name = std::string{n.ptr, n.len};
    ++fields;
    --len;
  }

  // Get the column width, which is the longest string length.
  auto const width = col.max_width();
  // Allocate.
  auto arr = PyArray_New(
    &PyArray_Type, 1, &len, NPY_STRING, nullptr, nullptr, width, 0, nullptr);
  auto base = (char const*) PyArray_DATA((PyArrayObject*) arr);

  Py_BEGIN_ALLOW_THREADS
  bzero((void*) base, width * len);
  for (; fields != col.end(); ++fields) {
    auto const field = *fields;
    memcpy((void*) (base + i++ * width), field.ptr, field.len);
  }
  Py_END_ALLOW_THREADS

  return {name, arr};
}


//------------------------------------------------------------------------------

StrArr
parse_str_arr(
  Column const& col,
  bool const header)
{
  auto len = col.size();

  size_t i = 0;
  auto fields = col.begin();

  std::string name;
  if (header) {
    if (len == 0)
      // Missing header.
      // FIXME: Check this earlier?
      return {};

    auto const& n = *fields;
    name = std::string{n.ptr, n.len};
    ++fields;
    --len;
  }

  if (len == 0)
    return StrArr{0, 0, {}, name};

  // Get the column width, which is the longest string length.
  auto const width = col.max_width();
  // Allocate.
  std::vector<char> chars(len * width, 0);
  char* base = chars.data();

  for (; fields != col.end(); ++fields) {
    auto const field = *fields;
    memcpy(base + i++ * width, field.ptr, field.len);
  }

  return {len, width, std::move(chars), name};
}


