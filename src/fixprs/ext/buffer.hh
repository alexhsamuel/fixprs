#include <iostream>

//------------------------------------------------------------------------------

struct Buffer
{
  char const* ptr;
  size_t len;
};


inline std::ostream&
operator<<(
  std::ostream& os,
  Buffer const& buf)
{
  for (char const* p = buf.ptr; p < buf.ptr + buf.len; ++p)
    std::cout << *p;
  return os;
}


