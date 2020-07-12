#include <cassert>
#include <string>

#include "buffer.hh"
#include "column.hh"

//------------------------------------------------------------------------------

struct Config
{
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

extern SplitResult split(Buffer buf, Config const& cfg);

//------------------------------------------------------------------------------

struct Array
{
  PyObject* array;
};


struct Result
{
  // Number of error values.
  size_t num_err = 0;
  // Location and value of the first error value.
  size_t err_idx = -1;
  std::string err_val;
};


extern void
parse_bytes(
  Column::Slice&& slice,
  Array&& const arr,
  Result& result);

//------------------------------------------------------------------------------

class Source
{
public:

  virtual ~Source();

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

  virutal ~BufferSource() {}

  virtual Buffer get_next() {
    return {
      buf_.ptr + pos_,
      std::min(chunk_, buf_.len - pos_),
    };
  }

  virtual void advance(size_t const len) {
    pos_ += len;
    assert(pos_ < buf_.len);
  }

private:

  Buffer const& buf_;
  size_t pos_;
  size_t const chunk_;

}


void process(Source& src, Config const& cfg) {
  std::vector<std::unique_ptr<Arr>> arrays;

  for (auto buf = src.get_next(); buf.len > 0; buf = src.get_next()) {
    auto split_result = split_columns(buf, cfg);
    src.advance(split_result.num_bytes);

    // Extend arrays.
    // - make sure there are enough of them
    // - make sure each is long enough

    for (size_t c = 0; c < split_result.cols.size(); ++c)
      // FIXME: Select columns to parse.
      results.push_back(pool.enqueue(parse, &cols[c], arrs[c]));

    for (auto&& result : results)

  }
}

