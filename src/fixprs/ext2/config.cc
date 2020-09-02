#include <cassert>
#include <Python.h>

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL FIXPRS_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

#include "config.hh"

//------------------------------------------------------------------------------

int
parse_column_config(
  PyObject* const dict,
  ColumnConfig* cfg)
{
  assert(dict != nullptr);

  auto dtype = PyDict_GetItemString(dict, "dtype");
  if (dtype == nullptr)
    cfg->descr = (PyObject*) PyArray_DescrFromType(NPY_STRING);
  else {
    if (PyArray_DescrConverter2(dtype, (PyArray_Descr**) &cfg->descr) != 0)
      return -1;
    if (cfg->descr == nullptr)
      cfg->descr = (PyObject*) PyArray_DescrFromType(NPY_STRING);
  }

  return 0;
}


int
parse_config(
  PyObject* const dict,
  Config* cfg)
{
  assert(dict != nullptr);

  auto col_cfg = PyDict_GetItemString(dict, "column");
  if (col_cfg != nullptr)
    if (parse_column_config(col_cfg, &cfg->col) != 0)
      return -1;

  return 0;
}




