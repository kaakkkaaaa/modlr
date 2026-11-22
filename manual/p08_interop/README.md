# Part 8: Python and C interop

Mojo does not live alone. It can call any Python library through the CPython interpreter, it can be imported by Python as a compiled extension module, and it can call C libraries (or be called by C) with no translation layer. This part shows all four directions, with the rules about types, ownership and errors that keep each boundary safe.

_Estimated time: 6–8 hours · Prerequisites: Parts 1–7_

Guide: [Part 8](https://claude.ai/artifact/76skpa3yrc1XGsrGXB4qf5#part-8)

## Goals

- Import Python modules (including NumPy and your own `.py` files) and work with `PythonObject` values.
- Convert values between Mojo and Python in both directions, and handle Python exceptions in Mojo.
- Export Mojo functions and structs to Python with `PythonModuleBuilder`, then import them with `mojo.importer` or a prebuilt `.so`.
- Call libc and your own C `.dylib`/`.so` with `external_call` and `OwnedDLHandle`, using correct C types, struct layouts, pointers and strings.
- Decide which side frees each piece of memory, and build a Mojo shared library a C program can link against.

## Steps

| Done | Step | Topic | My file |
|:---:|---|---|---|
| [ ] | 8.1 | The interop map | `s01_interop_map.mojo` |
| [ ] | 8.2 | Importing Python modules | `s02_importing_python_modules.mojo` |
| [ ] | 8.3 | PythonObject and Python collections | `s03_pythonobject_python_collections.mojo` |
| [ ] | 8.4 | Converting values in both directions | `s04_converting_values_both.mojo` |
| [ ] | 8.5 | Python exceptions in Mojo | `s05_python_exceptions_mojo.mojo` |
| [ ] | 8.6 | Using NumPy from Mojo | `s06_using_numpy_mojo.mojo` |
| [ ] | 8.7 | Calling Mojo from Python: your first extension module | `s07_calling_mojo_python.mojo` |
| [ ] | 8.8 | Binding Mojo types: constructors, methods and values | `s08_binding_mojo_types.mojo` |
| [ ] | 8.9 | Keyword and variadic arguments, building, and limits | `s09_keyword_variadic_arguments.mojo` |
| [ ] | 8.10 | Strategy: porting Python code to Mojo | `s10_strategy_porting_python.mojo` |
| [ ] | 8.11 | Calling C: types and external_call | `s11_calling_c_types.mojo` |
| [ ] | 8.12 | Shared libraries, structs and pointers | `s12_shared_libraries_structs.mojo` |
| [ ] | 8.13 | Strings and memory: who frees what | `s13_strings_memory_who.mojo` |
| [ ] | 8.14 | Calling Mojo from C, and a safety checklist | `s14_calling_mojo_c.mojo` |

File names are suggestions. Keep them as valid identifiers (letters, digits, `_`, no leading digit) so they stay importable.

## Notes

<!-- What clicked, what confused me, links. One subsection per step works well. -->

## Checkpoint

Answer in your own words before opening the answers in the guide.

1. Why does `var n = Int(obj)` fail for a `PythonObject`, and what do you write instead?

   > _My answer:_

2. You built a Mojo program that imports NumPy and copied the executable to a fresh machine. It fails at start-up. Why?

   > _My answer:_

3. In an extension module, why must `PyInit_mod` be `@export` and `abi("C")`, while registered functions can be ordinary `raises` functions?

   > _My answer:_

4. Write the module init for a module `geo` that exposes a struct `Point` with a no-argument initializer and a method `norm` using the pointer style.

   > _My answer:_

5. What goes wrong if you call `external_call["abs", c_int](-42)` with a bare literal instead of `c_int(-42)`?

   > _My answer:_

6. C's `strdup` returns a newly allocated `char*` or NULL. How do you declare the return type, read the string, and clean up?

   > _My answer:_

7. A C header declares `struct S { char a; double b; int c; };`. Write the Mojo mirror and give its size on a 64-bit platform.

   > _My answer:_

8. Your exported Mojo function uses `parallelize` and segfaults when called from a C program, but works from a Mojo `main()`. What is missing?

   > _My answer:_

## Manual pages

- [Python interoperability](https://mojolang.org/docs/manual/python/)
- [Calling Python from Mojo](https://mojolang.org/docs/manual/python/python-from-mojo/)
- [Python types](https://mojolang.org/docs/manual/python/types/)
- [Calling Mojo from Python](https://mojolang.org/docs/manual/python/mojo-from-python/)
- [Using Mojo's C foreign function interface](https://mojolang.org/docs/manual/c-ffi/)
- [Compilation targets (Call a Mojo shared library from C or C++)](https://mojolang.org/docs/tools/compilation/)
