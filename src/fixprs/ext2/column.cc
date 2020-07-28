#include "column.hh"


SplitResult
split_columns(
  Buffer const buf, 
  Config const& cfg)
{
  // FIXME: Take from cfg. 
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

  // Number of parsed lines.
  size_t size = 0;
  // Beginning of first unparsed line.
  size_t end = 0;

  size_t start = 0;
  bool escaped = false;

  for (size_t i = 0; i < buf.len; ++i) {
    char const c = buf.ptr[i];

    if (c == eol) {
      // End the field.
      cols.at(col_idx++).append(start, i, escaped);
      escaped = false;
      end = start = i + 1;

      // Remaining colums are missing in this row.
      for (; col_idx < cols.size(); ++col_idx)
        cols.at(col_idx).append_missing();

      ++size;
      assert(cols.at(0).size() == size);

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

  // FIXME: Handle the end of file somehow.

  // // Finish the current field.
  // if (start < buf.len)
  //   cols.at(col_idx++).append(start, buf.len, escaped);

  // // Remaining colums are missing in this row.
  // if (col_idx > 0)
  //   for (; col_idx < cols.size(); ++col_idx)
  //     cols.at(col_idx).append_missing();

  for (auto& col : cols)
    col.resize(size);

  result.num_bytes = end;
  result.num_rows = size;
  return result;
}


