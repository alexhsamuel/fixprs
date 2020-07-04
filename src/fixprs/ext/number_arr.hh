#pragma once

#include <experimental/optional>

#include "column.hh"
#include "fast_double_parser.h"

//------------------------------------------------------------------------------

inline bool oadd(uint64_t a, uint64_t b, uint64_t& r) { return __builtin_uaddll_overflow(a, b, &r); }
inline bool oadd( int64_t a,  int64_t b,  int64_t& r) { return __builtin_saddll_overflow(a, b, &r); }
inline bool omul(uint64_t a, uint64_t b, uint64_t& r) { return __builtin_umulll_overflow(a, b, &r); }
inline bool omul( int64_t a,  int64_t b,  int64_t& r) { return __builtin_smulll_overflow(a, b, &r); }

//------------------------------------------------------------------------------

template<class T>
struct NumberArr
{
  using vals_type = std::vector<T>;

  size_t len;
  T min;
  T max;
  vals_type vals;
  std::string name;
};


template<class T> extern inline optional<T> parse_number(Buffer const buf);

template<>
inline optional<uint64_t>
parse_number<uint64_t>(
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
parse_number<int64_t>(
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
parse_number<float64_t>(
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

template<class T, Parse<T> PARSE=parse_number<T>>
extern inline optional<NumberArr<T>>
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


