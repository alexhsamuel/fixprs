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

class Array
{
public:

  Array(
    size_t const width,
    size_t const len)
  : width_(width)
  , idx_(0)
  {
    arr_ = PyArray_New(
      &PyArray_Type, 1, &len, NPY_STRING, nullptr, nullptr, width, 0, nullptr);
    assert(arr_ != nullptr);  // FIXME
    ptr_ = (char*) PyArray_DATA((PyArrayObject*) arr_);
    stride_ = width;
  }

  ~Array() {
    if (arr_ != nullptr) {
      Py_DECREF(arr_);
      arr_ = nullptr;
    }
  }

  Array(Array const&) = delete;
  void operator=(Array const&) = delete;

  void expand(size_t const len) {
    assert(arr_ != nullptr);
    if (len > PyArray_Length(arr_))
      // FIXME: Resize.
      ;
  }

  PyObject* release(size_t const len) {
    assert(arr_ != nullptr);
    // FIXME: Resize.
    auto arr = arr_;
    arr_ = nullptr;
    ptr_ = nullptr;
    return arr;
  }

  Result parse(Column const& col) {
    for (
      auto field = col.begin(), size_t i = 0;
      field != col.end();
      ++field, ++i) {
      auto const ptr = ptr_ + idx_ * stride_;
      // Copy the bytes in.
      memcpy(ptr, field->ptr, field->len);
      // Zero out the rest of the field.
      memset(ptr + field->len, 0, width_ - field->len);
      // Advance.
      ++idx_;
    }

    return Result{};
  }

private:

  size_t width_;
  size_t idx_;

  PyObject* arr_;
  char* ptr_;
  size_t stride_;

};


struct Result
{
  // Number of error values.
  size_t num_err = 0;
  // Location and value of the first error value.
  size_t err_idx = -1;
  std::string err_val;
};


extern Result parse_bytes(Column const& col, Array& arr);

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


Result parse_bytes(Column const& col, Array& arr) {
  Result result;

  auto fields = col.begin();

  for (size_t i = 0; fields != col.end(); ++fields, ++i) {
    auto const field = *fields;
    auto const ptr = 
  }
}


void process(Source& src, Config const& cfg) {
  std::vector<Arr> arrays;

  for (auto buf = src.get_next(); buf.len > 0; buf = src.get_next()) {
    auto split_result = split_columns(buf, cfg);

    // Extend arrays.
    // - make sure there are enough of them

    // for (auto arr& : arrays)
    //   arr.expand(...);

    for (size_t i = 0; i < split_results.cols.size(); ++i) {
      auto result = arrays[i].parse(cols[i]);
      // FIXME: Handle result.
    }

    // for (size_t c = 0; c < split_result.cols.size(); ++c)
    //   // FIXME: Select columns to parse.
    //   results.push_back(pool.enqueue(parse, &cols[c], arrs[c]));

    // for (auto&& result : results)

    src.advance(split_result.num_bytes);
  }
}

