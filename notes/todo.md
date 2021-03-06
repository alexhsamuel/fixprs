# Current plan

- [x] factor out array management from parsing (to support structs)
- [ ] configure column dtypes/parsers
- [ ] clean up NumPy CAPI import
- [ ] template the value parser
- [ ] float parser
- [ ] bool parser
- [ ] str 'U' parser
- [ ] object str parser
- [ ] test cases with larger tables
- [ ] add float column

- [x] benchmark setup vs. NumPy, Pandas
- [x] put back threads

# Upcoming

- [ ] `parse()`
  - [ ] auto dtype
  - [ ] pass in dtypes
  - [ ] clean up `load_file()`

- [ ] string decoding
  - [ ] string arrays should return U dtype
  - [ ] integrate ASCII decoder
  - [ ] integrate UTF-8 decoder
  - [ ] general Python decoder
  
- [ ] fix `std::experimental::optional`

- [ ] parse to recarray


# Configuration

- Column names.
  - Given.
  - Determined from first row.
  - Enumerated.

- Column dtypes.
  - str
  - bytes
  - float64
  - int64
  - other integer types
  - datetime
  - date
  - bool (what choices?)
  - categorial
  - O array of str or bytes

- Error value.

- Skip columns: don't error-check, parse, or return

- Max rows

- Auto column dtype.
  - What types are possible, and what's the algorithm?

- How to handle problems.
  - Categories:
    - Missing or extra fields in a row.
    - Empty fields.
    - Fields that don't parse correctly.
  - Treatment:
    - fill / replacement value
    - mark rows / cols with problems
    - exception with detailed location

