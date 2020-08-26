#include "ThreadPool.h"

#include "array.hh"
#include "column.hh"
#include "config.hh"
#include "parse.hh"
#include "parse_int.hh"
#include "source.hh"

//------------------------------------------------------------------------------

struct ParseResult
{
  size_t num_resize = 0;
  size_t total_resize = 0;
};


struct ColResult
{
  // Number of error values.
  size_t num_err = 0;
  // Location and value of the first error value.
  size_t first_err_idx = -1;
  std::string first_err_val;
};


class Parser
{
public:

  virtual ~Parser() {}
  virtual ColResult parse(Column const& col, Target::Ptr ptr, size_t idx) = 0;

};


class BytesParser
: public Parser
{
public:

  BytesParser(
    size_t const width)
  : width_(width)
  {
  }

  ~BytesParser() {}

  ColResult parse(
    Column const& col,
    Target::Ptr const ptr,
    size_t r)
  {
    ColResult res;

    for (auto const field : col) {
      auto const len = std::min(field.len, width_);
      auto const p = ptr.first + r * ptr.second;

      // Copy the bytes in.
      memcpy(p, field.ptr, len);
      // Zero out the rest of the field.
      memset(p + len, 0, width_ - len);

      if (unlikely(field.len > width_)) {
        if (res.num_err++ == 0) {
          res.first_err_idx = r;
          res.first_err_val = std::string(field.ptr, field.len);
        }
      }

      // Advance.
      ++r;
    }

    return res;
  }

private:

  size_t const width_;

};


class Int64Parser
: public Parser
{
public:

  ~Int64Parser() {}

  ColResult parse(
    Column const& col,
    Target::Ptr const ptr,
    size_t r)
  {
    ColResult res;

    for (auto field : col) {
      auto const val = ::parse<int64_t>(field);
      if (likely(val)) {
        auto const p = (long*) (ptr.first + r * ptr.second);
        *p = *val;
      }
      else if (res.num_err++ == 0) {
        res.first_err_idx = r;
        res.first_err_val = std::string(field.ptr, field.len);
      }

      // Advance.
      ++r;
    }

    return res;
  }

};


PyObject*
parse_source(
  Source& src,
  Config const& cfg)
{
  ParseResult res;
  ArraysTarget target(0, cfg);
  std::vector<std::unique_ptr<Parser>> parsers;
  size_t r = 0;

  // FIXME: cfg threading, including no threading.
  ThreadPool pool{5};

  for (auto buf = src.get_next(); buf.len > 0; buf = src.get_next()) {
    auto split_result = split_columns(buf, cfg);
    std::cerr << "split "
              << split_result.num_bytes << " bytes, "
              << split_result.num_rows << " rows, "
              << split_result.cols.size() << " cols\n";

    // Add columns if necessary.
    while (unlikely(target.num_cols() < split_result.cols.size())) {
      auto const nc = target.num_cols();
      // FIXME: For testing only.
      if (nc == 0) {
        target.add_col(NPY_STRING, 32);
        parsers.emplace_back(std::make_unique<BytesParser>(32));
      }
      else {
        target.add_col(NPY_INT64);
        parsers.emplace_back(std::make_unique<Int64Parser>());
      }
    }
      
    auto const new_r = r + split_result.num_rows;
    if (target.check_size(new_r))
      res.num_resize += 1;

    std::vector<std::future<ColResult>> parse_results;
    for (size_t c = 0; c < split_result.cols.size(); ++c) {
      auto& col = split_result.cols[c];
      auto parser = parsers[c].get();
      parse_results.push_back(pool.enqueue([&col, parser, &target, c, r] {
        return parser->parse(col, target.get_pointer(c), r);
      }));
    }
    // FIXME: What if there are fewer split cols than cols in the target?

    for (auto&& result : parse_results) {
      // FIXME: Do something real with results.
      auto r = result.get();
      if (r.num_err > 0)
        std::cerr << "ERRORS: count: " << r.num_err
                  << " first idx: " << r.first_err_idx
                  << " first val: " << r.first_err_val
                  << "\n";
    }

    src.advance(split_result.num_bytes);
    r = new_r;

    // FIXME: Can we split the next chunk while still parsing this one?
  }

  std::cerr << "NUM RESIZE: " << res.num_resize << "\n";
  return target.release(r);
}


