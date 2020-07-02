#include "column.hh"


std::vector<Column>
split_columns(
  Buffer const buffer,
  char const sep,
  char const eol,
  char const quote)
{
  if (buffer.len == 0)
    return {};

  std::vector<Column> cols;
  // FIXME: Replace col_idx with an iterator?
  std::vector<Column>::size_type col_idx = 0;
  cols.emplace_back(buffer);

  size_t start = 0;
  bool escaped = false;

  for (size_t i = 0; i < buffer.len; ++i) {
    char const c = buffer.ptr[i];

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
        cols.emplace_back(buffer, cols[0].size() - 1);
    }
    else if (c == quote) {
      // Fast-forward through quoted strings: skip over the opening quote, and
      // copy characters until the closing quote.
      escaped = true;
      for (++i; i < buffer.len && buffer.ptr[i] != quote; ++i)
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
  if (start < buffer.len)
    cols.at(col_idx++).append(start, buffer.len, escaped);

  // Remaining colums are missing in this row.
  if (col_idx > 0)
    for (; col_idx < cols.size(); ++col_idx)
      cols.at(col_idx).append_missing();

  return cols;
}


