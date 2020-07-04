#include <cstdlib>
#include <string>
#include <vector>

#include "str_arr.hh"

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


