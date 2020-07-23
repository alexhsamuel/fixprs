#include "column.hh"


SplitResult
split_columns(
  Buffer const buf, 
  Config const& cfg)
{
  // FIXME
  char const sep = ',';
  char const eol = '\n';
  char const quote = '"';

  if (buf.len == 0)
    return {};

  SplitResult result;
  std::vector<Column>& cols = result.cols;  // FIXME

  // FIXME: Replace col_idx with an iterator?
  std::vector<Column>::size_type col_idx = 0;
  cols.emplace_back(buf);

  size_t start = 0;
  bool escaped = false;

  for (size_t i = 0; i < buf.len; ++i) {
    char const c = buf.ptr[i];

    if (c == eol) {
      // End the field.
      cols.at(col_idx++).append(start, i, escaped);
      escaped = false;
      start = i + 1;

      // Remaining colums are missing in this row.
      for (; col_idx < cols.size(); ++col_idx)
        cols.at(col_idx).append_missing();

      // End the line.
      col_idx = 0;
    }
    else if (c == sep) {
      // End the field.
      cols.at(col_idx++).append(start, i, escaped);
      escaped = false;
      start = i + 1;

      if (col_idx >= cols.size())
        // Create a new column, and back-fill it with missing.
        cols.emplace_back(buf, cols[0].size() - 1);
    }
    else if (c == quote) {
      // Fast-forward through quoted strings: skip over the opening quote, and
      // copy characters until the closing quote.
      escaped = true;
      for (++i; i < buf.len && buf.ptr[i] != quote; ++i)
        ;
      // FIXME: else: unclosed quote.
    }
    else
      // Normal character.
      // FIXME: Check for escape characters.
      ;

    // FIXME: Trailing field?
  }

  // Finish the current field.
  if (start < buf.len)
    cols.at(col_idx++).append(start, buf.len, escaped);

  // Remaining colums are missing in this row.
  if (col_idx > 0)
    for (; col_idx < cols.size(); ++col_idx)
      cols.at(col_idx).append_missing();

  result.num_bytes = buf.len;
  result.num_rows = cols[0].size();
  return result;
}


#if 0

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

#endif
