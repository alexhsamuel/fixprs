#include <memory>
#include <vector>

#include "array.hh"
#include "config.hh"
#include "source.hh"

//------------------------------------------------------------------------------

using ArrayVec = std::vector<std::unique_ptr<Array>>;

extern ArrayVec parse_source(Source& src, Config const& cfg);

