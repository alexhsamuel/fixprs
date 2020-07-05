#pragma once

#include <string>
#include <vector>

#include <Python.h>

#include "column.hh"

//------------------------------------------------------------------------------

extern std::pair<std::string, PyObject*>
parse_bytes_col(
  Column const& col);

//------------------------------------------------------------------------------
// FIXME: Deprecated

struct StrArr
{
  size_t len;
  size_t width;
  std::vector<char> chars;
  std::string name;
};

extern StrArr
parse_str_arr(
  Column const& col,
  bool const header=true);

