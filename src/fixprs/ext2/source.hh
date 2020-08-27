#pragma once

#include <cassert>

#include "buffer.hh"

//------------------------------------------------------------------------------

class Source
{
public:

  virtual ~Source() {}

  virtual Buffer get_next() = 0;
  virtual void advance(size_t) = 0;

};


class BufferSource
: public Source
{
public:

  BufferSource(Buffer const& buf, size_t const chunk)
  : buf_(buf),
    pos_(0),
    chunk_(chunk)
  {
  }

  virtual ~BufferSource() {}

  virtual Buffer get_next() {
    return {
      buf_.ptr + pos_,
      std::min(chunk_, buf_.len - pos_),
    };
  }

  virtual void advance(size_t const len) {
    pos_ += len;
    assert(pos_ <= buf_.len);
  }

private:

  Buffer const& buf_;
  size_t pos_;
  size_t const chunk_;

};


