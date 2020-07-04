#pragma once

#include <string>
#include <vector>

#include "column.hh"

//------------------------------------------------------------------------------

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

