#include "ThreadPool.h"

#include "array.hh"
#include "column.hh"
#include "config.hh"
#include "source.hh"

//------------------------------------------------------------------------------

std::vector<Array> parse(Source& src, Config const& cfg)
{
  std::vector<Array> arrays;

  // FIXME: cfg threading, including no threading.
  ThreadPool pool{5};

  for (auto buf = src.get_next(); buf.len > 0; buf = src.get_next()) {
    auto split_result = split_columns(buf, cfg);
    std::cerr << "split "
              << split_result.num_bytes << " bytes, "
              << split_result.num_rows << " rows, "
              << split_result.cols.size() << " cols\n";

    // Extend arrays.
    while (arrays.size() < split_result.cols.size())
      // FIXME
      arrays.emplace_back(32, cfg.initial_column_len);

    std::vector<std::future<Result>> parse_results;
    for (size_t i = 0; i < split_result.cols.size(); ++i) {
      auto& arr = arrays[i];
      arr.expand(arr.size() + split_result.num_rows);
      auto& col = split_result.cols[i];
      parse_results.push_back(pool.enqueue([&] { return arr.parse(col); }));
    }

    for (auto&& result : parse_results)
      // FIXME: Do something with them.
      result.get();

    // for (size_t i = 0; i < split_results.cols.size(); ++i) {
    //   arr.expand(...);
    //
    //   auto result = arrays[i].parse(cols[i]);
    //   // FIXME: Handle result.
    // }

    // for (size_t c = 0; c < split_result.cols.size(); ++c)
    //   // FIXME: Select columns to parse.
    //   results.push_back(pool.enqueue(parse, &cols[c], arrs[c]));

    // for (auto&& result : results)

    src.advance(split_result.num_bytes);
  }

  return arrays;
}


