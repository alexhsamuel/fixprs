#include <assert.h>
#include <fcntl.h>
#include <Python.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL FIXPRS_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

#include "column.hh"
#include "source.hh"
#include "parse.hh"

//------------------------------------------------------------------------------

extern "C" {

static PyObject*
fn_split_columns(
  PyObject* const self,
  PyObject* const args,
  PyObject* const kw_args)
{
  static char const* const keywords[] = {"obj", NULL};
  PyObject* obj;

  if (!PyArg_ParseTupleAndKeywords(args, kw_args, "O", (char**) keywords, &obj))
    return NULL;

  auto const memview = PyMemoryView_FromObject(obj);
  if (memview == nullptr) {
    Py_DECREF(obj);
    return nullptr;
  }

  auto const pybuf = PyMemoryView_GET_BUFFER(memview);
  Buffer buf{static_cast<char const*>(pybuf->buf), (size_t) pybuf->len};
  
  Config cfg;
  auto result = split_columns(buf, cfg);
  std::cerr << "split "
            << result.num_bytes << " bytes, "
            << result.num_rows << " rows, "
            << result.cols.size() << " cols\n";

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject*
fn_parse_buffer(
  PyObject* const self,
  PyObject* const args,
  PyObject* const kw_args)
{
  static char const* const keywords[] = {"obj", "cfg", nullptr};
  PyObject* obj;
  PyObject* cfg_dict = nullptr;

  if (!PyArg_ParseTupleAndKeywords(
        args, kw_args, "O|O!", (char**) keywords,
        &obj, &PyDict_Type, cfg_dict))
    return nullptr;

  Config cfg;
  if (cfg_dict != nullptr && parse_config(cfg_dict, &cfg) != 0)
    return nullptr;

  auto const memview = PyMemoryView_FromObject(obj);
  if (memview == nullptr)
    return nullptr;

  auto const pybuf = PyMemoryView_GET_BUFFER(memview);
  Buffer buf{static_cast<char const*>(pybuf->buf), (size_t) pybuf->len};
  BufferSource source{buf, cfg.chunk_size};
  auto res = parse_source(source, cfg);
  
  Py_DECREF(memview);

  return res;
}


static PyMethodDef methods[] = {
  {"split_columns", (PyCFunction) fn_split_columns, METH_VARARGS | METH_KEYWORDS, NULL},
  {"parse_buffer", (PyCFunction) fn_parse_buffer, METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef
module_def = {
  PyModuleDef_HEAD_INIT,
  "fixprs.ext2",  
  NULL,
  -1,      
  methods,
};


PyMODINIT_FUNC
PyInit_ext2(void)
{
  _import_array();

  PyObject* module = PyModule_Create(&module_def);
  assert(module != NULL);
  return module;
}


}  // extern "C"

