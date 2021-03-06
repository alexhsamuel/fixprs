#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL FIXPRS_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

#include "ThreadPool.h"

#include "array.hh"
#include "column.hh"
#include "config.hh"
#include "parse.hh"
#include "parse_int.hh"
#include "source.hh"

#define DEBUG_PRINT false

//------------------------------------------------------------------------------

// A vector of managed heap-allocated objects.
template<class T>
using PtrVec = std::vector<std::unique_ptr<T>>;

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


size_t
choose_new_size(
  size_t const len,
  size_t const new_len,
  ResizeConfig const& cfg)
{
  if (cfg.grow) {
    auto l = len;
    while (l < new_len)
      l = std::max(
        (size_t) (l * cfg.grow_factor),
        l + cfg.min_grow);
    return l;
  }
  else
    // FIXME
    abort();
}


/*
 * Adds a column to `target` and a corresponding parser to `parsers`.
 */
void
add_column(
  ColumnConfig const& cfg,
  Target& target,
  PtrVec<Parser>& parsers)
{
  PyArray_Descr* descr = (PyArray_Descr*) cfg.descr;

  // FIXME: This is not the right place.
  if (descr == nullptr) {
    descr = PyArray_DescrNewFromType(NPY_STRING);
    descr->elsize = 32;
  }

  target.add_col((PyObject*) descr);
  if (descr->kind == 'S')
    parsers.emplace_back(std::make_unique<BytesParser>(descr->elsize));
  else if (descr->kind == 'i' && descr->elsize == 8)
    parsers.emplace_back(std::make_unique<Int64Parser>());
  else
    abort();
}


PyObject*
parse_source(
  Source& src,
  Config const& cfg)
{
  ParseResult res;

  // The output target for parsed data, and a parallel vector of parsers.
  std::unique_ptr<Target> target = std::make_unique<ArraysTarget>(0);
  PtrVec<Parser> parsers;

  size_t r = 0;

  // FIXME: cfg threading, including no threading.
  ThreadPool pool{5};

  for (auto buf = src.get_next(); buf.len > 0; buf = src.get_next()) {
    auto split_result = split_columns(buf, cfg);
    if (DEBUG_PRINT)
      std::cerr << "split "
                << split_result.num_bytes << " bytes, "
                << split_result.num_rows << " rows, "
                << split_result.cols.size() << " cols\n";

    // Add columns if necessary.
    while (unlikely(target->num_cols() < split_result.cols.size()))
      add_column(cfg.col, *target.get(), parsers);
    assert(target->num_cols() == parsers.size());
      
    // Resize target columns, if this batch doesn't fit.
    auto const new_r = r + split_result.num_rows;
    if (unlikely(target->length() < new_r)) {
      target->resize(choose_new_size(target->length(), new_r, cfg.resize));
      ++res.num_resize;
    }

    std::vector<std::future<ColResult>> parse_results;
    for (size_t c = 0; c < split_result.cols.size(); ++c) {
      auto& col = split_result.cols[c];
      auto parser = parsers[c].get();
      parse_results.push_back(pool.enqueue([&col, parser, &target, c, r] {
        return parser->parse(col, target->get_pointer(c), r);
      }));
    }
    // FIXME: What if there are fewer split cols than cols in the target?

    for (auto&& result : parse_results) {
      // FIXME: Do something real with results.
      auto r = result.get();
      if (r.num_err > 0)
        if (DEBUG_PRINT)
          std::cerr << "ERRORS: count: " << r.num_err
                    << " first idx: " << r.first_err_idx
                    << " first val: " << r.first_err_val
                    << "\n";
    }

    src.advance(split_result.num_bytes);
    r = new_r;

    // FIXME: Can we split the next chunk while still parsing this one?
  }

  if (target->length() != r) {
    target->resize(r);
    ++res.num_resize;
  }

  if (DEBUG_PRINT)
    std::cerr << "NUM RESIZE: " << res.num_resize << "\n";
  return target->release();
}


