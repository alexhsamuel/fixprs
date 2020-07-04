#include <cassert>
#include <cstdlib>
#include <cstring>

#include "fast_double_parser.h"

#include "column.hh"
#include "csv2.hh"
#include "str_arr.hh"

// FIXME: UTF-8 (and other encodings?)

//------------------------------------------------------------------------------

inline bool oadd(uint64_t a, uint64_t b, uint64_t& r) { return __builtin_uaddll_overflow(a, b, &r); }
inline bool oadd( int64_t a,  int64_t b,  int64_t& r) { return __builtin_saddll_overflow(a, b, &r); }
inline bool omul(uint64_t a, uint64_t b, uint64_t& r) { return __builtin_umulll_overflow(a, b, &r); }
inline bool omul( int64_t a,  int64_t b,  int64_t& r) { return __builtin_smulll_overflow(a, b, &r); }

template<class T> inline optional<T> parse(Buffer const buf);

template<>
inline optional<uint64_t>
parse<uint64_t>(
  Buffer const buf)
{
  if (buf.len == 0)
    return {};

  // FIXME: Accept leading +?

  uint64_t val = 0;
  for (auto p = buf.ptr; p < buf.ptr + buf.len; ++p) {
    auto const c = *p;
    if ('0' <= c && c <= '9') {
      if (omul(val, 10, val) || oadd(val, c - '0', val))
        return {};
    }
    else
      return {};
  }

  return val;
}


template<>
inline optional<int64_t>
parse<int64_t>(
  Buffer const buf)
{
  if (buf.len == 0)
    // Empty string.
    return {};

  auto p = buf.ptr;
  auto const end = buf.ptr + buf.len;
  bool negative = false;
  if (p[0] == '-') {
    negative = true;
    ++p;
  }
  else if (p[0] == '+')
    ++p;

  if (p == end)
    // No digits.
    return {};

  int64_t val = 0;
  while (p < end) {
    auto const c = *p++;
    if ('0' <= c && c <= '9') {
      if (omul(val, 10, val) || oadd(val, negative ? '0' - c : c - '0', val))
        // Overflow.
        return {};
    }
    else
      // Not a digit.
      return {};
  }

  return val;
}


template<>
inline optional<float64_t>
parse<float64_t>(
  Buffer const buf)
{
  if (buf.len == 0)
    // Empty string.
    return {};

  double val;
  if (fast_double_parser::parse_number(buf.ptr, &val))
    return val;
  else
    return {};
}


//------------------------------------------------------------------------------

template<class T> using Parse = optional<T>(*)(Buffer);

template<class T, Parse<T> PARSE=parse<T>>
inline optional<NumberArr<T>>
parse_number_arr(
  Column const& col,
  bool const header=true)
{
  auto len = col.size();
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
    return NumberArr<T>{0, 0, 0, {}, name};

  // Parse the first value.
  auto const val = PARSE(*fields);
  if (!val)
    return {};
  ++fields;

  // Initialize min and max.
  T min = *val;
  T max = *val;

  // Allocate space for results.
  typename NumberArr<T>::vals_type vals;
  vals.reserve(len);
  vals.push_back(*val);

  for (; fields != col.end(); ++fields) {
    auto const& field = *fields;
    auto const val = PARSE(field);
    if (val) {
      vals.push_back(*val);
      if (*val < min)
        min = *val;
      if (*val > max)
        max = *val;
    }
    else
      return {};
  }

  assert(vals.size() == len);
  return NumberArr<T>{len, min, max, std::move(vals), name};
}


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


