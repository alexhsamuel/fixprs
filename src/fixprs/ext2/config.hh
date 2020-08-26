#pragma once

//------------------------------------------------------------------------------

/*
 * Configuration for how an array is dynamically resized.
 */
struct ResizeConfig
{
  /* Whether to grow the array if necessary when adding elements.  */
  bool grow = true;

  size_t initial_size = 0;

  /* Resize factor by which to grow.  */
  double grow_factor = 2;

  /* Minimum number of items to grow.  */
  size_t min_grow = 1;

};


struct Config
{
  /* Size in bytes of chunk to process at one time from source. */
  size_t chunk_size = 64 * 1024 * 1024;

  /*
   * Initial size for new columns.  If the exact length of the table is known,
   * set this to the full length.  If the approximate length is known, set this
   * to a bit more, to avoid reallocation.
   */
  size_t initial_column_len = 0;

  ResizeConfig resize;

};


