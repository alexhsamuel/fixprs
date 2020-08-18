#include "ThreadPool.h"

#include "array.hh"
#include "column.hh"
#include "config.hh"
#include "parse.hh"
#include "source.hh"

//------------------------------------------------------------------------------

ArrayVec parse(Source& src, Config const& cfg)
{
  ArrayVec arrs;

  // FIXME: cfg threading, including no threading.
  ThreadPool pool{5};

  for (auto buf = src.get_next(); buf.len > 0; buf = src.get_next()) {
    auto split_result = split_columns(buf, cfg);
    std::cerr << "split "
              << split_result.num_bytes << " bytes, "
              << split_result.num_rows << " rows, "
              << split_result.cols.size() << " cols\n";

    // Extend arrs.
    while (arrs.size() < split_result.cols.size())
      // FIXME: Empty overhang.
      arrs.emplace_back(std::make_unique<BytesArray>(cfg.initial_column_len, 32));

    std::vector<std::future<Result>> parse_results;
    for (size_t i = 0; i < split_result.cols.size(); ++i) {
      auto& arr = arrs[i];
      arr->expand(arr->size() + split_result.num_rows);
      auto& col = split_result.cols[i];
      parse_results.push_back(pool.enqueue([&] { return arr->parse(col); }));
    }

    for (auto&& result : parse_results)
      // FIXME: Do something with results.
      result.get();

    src.advance(split_result.num_bytes);

    // FIXME: Can we split the next chunk while still parsing this one?
  }

  return arrs;
}


