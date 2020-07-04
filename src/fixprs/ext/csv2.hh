#include <cassert>
#include <experimental/optional>
#include <iostream>
#include <string>
#include <vector>

using std::experimental::optional;

// FIXME: static assert that this is right.
using float64_t = double;

#include "number_arr.hh"
#include "str_arr.hh"

//------------------------------------------------------------------------------

// FIXME: Use a respectable variant.

class Array
{
public:

  using variant_type = enum {
    VARIANT_INT,
    VARIANT_FLOAT,
    VARIANT_STRING,
  };

  Array(NumberArr<int64_t>&& arr) : variant_{VARIANT_INT}, int_arr_{std::move(arr)} {}
  Array(NumberArr<float64_t>&& arr) : variant_{VARIANT_FLOAT}, float_arr_{std::move(arr)} {}
  Array(StrArr&& arr) : variant_{VARIANT_STRING}, str_arr_{std::move(arr)} {}

  variant_type variant() const { return variant_; }

  NumberArr<int64_t> const& int_arr() const { return *int_arr_; }
  NumberArr<float64_t> const& float_arr() const { return *float_arr_; }
  StrArr const& str_arr() const { return *str_arr_; }

private:

  variant_type variant_;

  optional<NumberArr<int64_t>> const int_arr_;
  optional<NumberArr<float64_t>> const float_arr_;
  optional<StrArr> const str_arr_;

};


extern void
print(
  std::ostream& os,
  Array const& arr,
  bool const print_values=true);


inline std::ostream&
operator<<(
  std::ostream& os,
  Array const& arr)
{
  print(os, arr, false);
  return os;
}


extern Array
parse_array_auto(
  Column const* col,
  bool header=true);


