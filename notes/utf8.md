See: 

This:
- https://bjoern.hoehrmann.de/utf-8/decoder/dfa/

Benchmark values show that Python's implementation is competitive?  But we don't
want to use CAPI endpoint, which all deal with PyObject*.  See
`Objects/unicodeobject.c:unicode_decode_utf8` for implementation.

Perhaps use one of Hoehrmann's variations, with ASCII fast path for prefix?

Also:
- https://github.com/bdonlan/branchless-utf8/commit/3802d3b0e10ea16810dd40f8116243971ff7603d
- https://github.com/KWillets/fastdecode-utf-8

