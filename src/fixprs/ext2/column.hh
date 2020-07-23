#pragma once

#include <cassert>
#include <vector>

#include "buffer.hh"
#include "config.hh"

//------------------------------------------------------------------------------

class Column
{
private:

  // FIXME: Use a more efficient encoding.
  struct Field
  {
    Field() : missing{true} {};

    Field(
      size_t start_,
      size_t len_,
      bool escaped_) 
    : start{start_}
    , len{len_}
    , escaped{escaped_}
    , missing{false} 
    {
    }

    unsigned long   start   : 46;  // 64 TB limit on total size
    unsigned long   len     : 16;  // 64 KB limit on field size
    bool            escaped :  1;
    bool            missing :  1;
  };

  static_assert(sizeof(Field) == 8, "wrong sizeof(Field)");

public:

  Column(
    Buffer const buf, 
    size_t const num_missing=0)
  : buf_(buf)
  , fields_(num_missing, Field{})
  , has_missing_(num_missing > 0)
  {
  }

  Column(Column&&) = default;

  size_t    size()          const { return fields_.size(); }
  bool      has_missing()   const { return has_missing_; }
  bool      has_empty()     const { return has_empty_; }
  size_t    max_width()     const { return max_width_; }

  void append(
    size_t const start,
    size_t const end,
    bool const escaped)
  {
    assert(0 <= start && start < buf_.len);
    assert(start <= end && end <= buf_.len);

    auto const len = end - start;
    fields_.emplace_back(start, len, escaped);

    if (len == 0)
      has_empty_ = true;
    if (len > max_width_)
      max_width_ = len;
  }

  void 
  append_missing()
  {
    fields_.emplace_back();
    has_missing_ = true;
  }

  class Iterator
  {
  public:

    Iterator(
      Column const& col, 
      size_t idx=0) 
    : col_(col),
      idx_(idx)
    {
      assert(0 <= idx_ && idx_ <= col_.size());
    }

    bool operator==(Iterator const& it) const { return it.idx_ == idx_; }
    bool operator!=(Iterator const& it) const { return it.idx_ != idx_; }
    void operator++() { ++idx_; }

    Buffer
    operator*() 
      const
    {
      auto const& field = col_.fields_.at(idx_);
      if (field.missing)
        return {nullptr, 0};
      else
        return {col_.buf_.ptr + field.start, field.len};
    }

  private:

    Column const& col_;
    size_t idx_;

  };

  Iterator begin() const { return Iterator(*this, 0); }
  Iterator end() const { return Iterator(*this, size()); }

private:

  Buffer const buf_;
  std::vector<Field> fields_;
  bool has_missing_ = false;
  bool has_empty_ = false;
  size_t max_width_ = 0;

};


//------------------------------------------------------------------------------

struct SplitResult
{
  // Number of bytes consumed.
  size_t num_bytes;

  // Number of rows.
  size_t num_rows;

  // Columns.
  std::vector<Column> cols;
};


extern SplitResult split_columns(Buffer buf, Config const& cfg);

