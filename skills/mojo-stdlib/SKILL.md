---
name: mojo-stdlib
---

# Mojo standard library

Do not invent a signature. Fetch `https://mojolang.org/docs/std/<path>.md`, or search <https://mojolang.org/llms-stdlib.txt> and open the page it names. How to read a signature: <https://mojolang.org/docs/api-docs.md>.

`std.gpu` is not in this library. GPU types live in `max.gpu`. See `mojo-gpu`.

## Imports

```mojo
from std.testing import assert_equal, TestSuite
from std.math import min, max, sqrt, ceildiv
from std.os import getenv
from std.memory.alloc import alloc, dealloc, Layout
```

Every explicit stdlib import uses the `std.` prefix.

Prelude, already in scope: `Int`, `UInt`, sized integers, `Float16`, `Float32`, `Float64`, `Bool`, `String`, `List`, `Dict`, `Optional`, `SIMD`, `DType`, `Error`, `Span`, `Pointer`, `UnsafePointer`, `print`, `range`, `len`, `rebind`, `Copyable`, `Movable`, `Writable`, `Equatable`, `Hashable`.

## Reading a signature

Parameters are compile-time and sit in `[]`. Arguments are runtime and sit in `()`.

| Marker | Meaning |
|---|---|
| `//` | Infer-only parameters. Callers omit them. |
| `/` | Positional-only. |
| `*` | Keyword-only. |

An unmarked argument is an immutable borrow. The other conventions are `mut`, `var`, `ref`, `out`, and `deinit`. `Self.T` is a parameter of the enclosing type. `self` is the instance and is not passed at the call. `def(Int) -> Int` used as a type is a function type, not a declaration.

## Packages

| Package | Use for |
|---|---|
| `algorithm` | `vectorize`, `map`, `tile` over index ranges |
| `atomic` | `Atomic`, `Ordering`, `fence` |
| `base64` | `b64encode`, `b64decode`, base16 |
| `benchmark` | `Bench`, `QuickBench`, `black_box`, `keep` |
| `bit` | `pop_count`, rotates, `next_power_of_two`, `log2_floor` |
| `collections` | `Array`, `Set`, `Deque`, other collection types |
| `compile` | Runtime compilation, assembly, IR |
| `complex` | Complex SIMD and scalars |
| `ffi` | C ABI: `c_int`, `DLHandle`, external calls |
| `format` | Text formatting traits |
| `hash` | Hash algorithms |
| `io` | `print`, files, `Writer` |
| `iter` | `enumerate`, `zip`, `map`, iterator traits |
| `itertools` | Lazy iterator tools |
| `logger` | Severity logging |
| `math` | `min`, `max`, `sqrt`, `ceildiv`, trig, `inf`, `nan` |
| `memory` | `Pointer`, `alloc`, `dealloc`, `ArcPointer`, `OwnedPointer` |
| `os` | `getenv`, filesystem, process |
| `pathlib` | Path joins and queries |
| `python` | Python interop. Use `mojo-python-interop` |
| `random` | Uniform and normal RNG |
| `reflection` | Compile-time type and function introspection |
| `runtime` | Worker threads. There is no `async` / `await` |
| `subprocess` | Run a process |
| `sys` | `has_accelerator`, target info, `exit` |
| `tempfile` | Temporary files and directories |
| `testing` | `assert_equal`, `assert_raises`, `TestSuite` |
| `time` | `perf_counter_ns`, `sleep`, `time_function` |
| `utils` | `Variant`, `StaticTuple`, indexing helpers |

`std.simd` is a module. Prefer the prelude `SIMD` type.

## Calls that usually come out wrong

```mojo
from std.math import min, max          # free functions, not methods

s.byte_length()                        # len(s) does not compile
s.count_codepoints()
var ch = String(s[byte=i])            # s[byte=i] is a StringSpan

ptr[unsafe_offset=i]
ptr.unsafe_write(value)

opt.or_else(default)
scores.get("missing", 0)              # scores["missing"] raises
var home = getenv("HOME")             # Optional[String]

var v = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
_ = v.cast[.uint32]()
_ = v.reduce_add()
```

`List[Int](1, 2, 3)` is not a constructor. A bare `[1, 2, 3]` is an `Array`.

## Tests

There is no `mojo test` subcommand. Run a file that discovers tests:

```mojo
from std.testing import assert_equal, assert_raises, TestSuite

def test_add() raises:
    assert_equal(1 + 1, 2)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

```bash
mojo run test_add.mojo
```

## Benchmarks

Keep results alive, or the compiler deletes the work:

```mojo
from std.benchmark import Bench, BenchConfig, keep, black_box
```

Fetch <https://mojolang.org/docs/std/benchmark.md> for the current `Bench` or `QuickBench` call before writing a harness.

## More calls

```mojo
from std.math import ceildiv, sqrt, iota
from std.algorithm import vectorize
from std.pathlib import Path
from std.os import getenv, abort
from std.ffi import c_int, DLHandle

var blocks = ceildiv(n, tile)                 # integer ceil division
var root = sqrt(Float64(n))
var home = getenv("HOME")                     # Optional[String]
var path = Path("data") / "input.txt"

def body(i: Int):
    data[i] = data[i] * 2

vectorize[simd_width](n, body)                # runtime-arg overload
```

`iota` fills a contiguous buffer with `0, 1, 2, ...`. `ceildiv` is the grid-size helper. `vectorize` steps by `simd_width` and passes the tail as a shorter SIMD. `min` and `max` in `std.math` accept SIMD and scalars of the same type.

`c_int` is 32 bits on every target. A C `long` is not `Int64` on every ABI. Use the alias in `std.ffi` that names the C type. `DLHandle` loads a shared library. Strings that cross the FFI boundary need an explicit NUL-terminated view. Fetch `std.ffi` before writing the call.

`Path` joins with `/` or `joinpath`. `exists`, `read_text`, and `write_text` are methods on `Path`.

`abort(message)` terminates. Use it when a function cannot `raises`, such as an `abi("C")` entry point.
