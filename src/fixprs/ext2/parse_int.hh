#pragma once

#include <optional>

#include "buffer.hh"

// FIXME: Elsewhere.
template<class T> inline std::optional<T> parse(Buffer const buf);

//------------------------------------------------------------------------------

inline bool oadd(uint64_t a, uint64_t b, uint64_t& r) { return __builtin_uaddl_overflow(a, b, &r); }
inline bool oadd( int64_t a,  int64_t b,  int64_t& r) { return __builtin_saddl_overflow(a, b, &r); }
inline bool omul(uint64_t a, uint64_t b, uint64_t& r) { return __builtin_umull_overflow(a, b, &r); }
inline bool omul( int64_t a,  int64_t b,  int64_t& r) { return __builtin_smull_overflow(a, b, &r); }

template<>
inline std::optional<uint64_t>
parse<uint64_t>(
  Buffer const buf)
{
  if (buf.len == 0)
    return {};

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
inline std::optional<int64_t>
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


