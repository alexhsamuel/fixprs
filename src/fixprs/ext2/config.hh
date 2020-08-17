#pragma once

//------------------------------------------------------------------------------

struct Config
{
  // Size in bytes of chunk to process at one time from source.
  size_t chunk_size = 64 * 1024 * 1024;

  // Initial size for new columns.  If the exact length of the table is known,
  // set this to the full length.  If the approximate lengt is known, set this
  // to a bit more, to avoid reallocation.
  size_t initial_column_len = 0;
};

