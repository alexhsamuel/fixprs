#include <cassert>
#include <cstdlib>
#include <cstring>

#include "column.hh"
#include "csv2.hh"
#include "str_arr.hh"

//------------------------------------------------------------------------------

Array
parse_array_auto(
  Column const* col,
  bool header)
{
  {
    auto arr = parse_number_arr<int64_t>(*col);
    if (arr)
      return {std::move(*arr)};
  }
  {
    auto arr = parse_number_arr<float64_t>(*col);
    if (arr)
      return {std::move(*arr)};
  }
  
  return {parse_str_arr(*col)};
}


//------------------------------------------------------------------------------

void
print(
  std::ostream& os,
  Array const& arr,
  bool const print_values)
{
  if (arr.variant() == Array::VARIANT_INT) {
    auto const& int_arr = arr.int_arr();
    os << "int64 column '" << int_arr.name 
       << "' len=" << int_arr.len
       << " min=" << int_arr.min << " max=" << int_arr.max << "\n";
    if (print_values)
      for (size_t i = 0; i < int_arr.len; ++i)
        os << i << '.' << ' ' << int_arr.vals[i] << '\n';
  }

  else if (arr.variant() == Array::VARIANT_FLOAT) {
    auto const& float_arr = arr.float_arr();
    os << "float64 column '" << float_arr.name 
       << "' len=" << float_arr.len
       << " min=" << float_arr.min
       << " max=" << float_arr.max << "\n";
    if (print_values)
      for (size_t i = 0; i < float_arr.len; ++i)
        os << i << '.' << ' ' << float_arr.vals[i] << '\n';
  }

  else if (arr.variant() == Array::VARIANT_STRING) {
    auto const& str_arr = arr.str_arr();
    os << "str column len=" << str_arr.len
       << " width=" << str_arr.width << '\n';
    if (print_values)
      for (size_t i = 0; i < str_arr.len; ++i) {
        os << i << '.' << ' ' << '[';
        char const* base = str_arr.chars.data();
        for (char const* p = base + i * str_arr.width;
             p < base + (i + 1) * str_arr.width;
             ++p)
          if (*p == 0)
            os << "·";
          else
            os << *p;
        os << ']' << '\n';
      }
  }
}


