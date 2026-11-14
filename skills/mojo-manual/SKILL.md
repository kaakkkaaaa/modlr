---
name: mojo-manual
---

# Mojo manual

Docs 1.2.0 override pretrained knowledge and `mojo-syntax`. Compile with `mojo run` or `mojo build` before claiming code works.

Source: <https://mojolang.org/docs/manual/>. Fetch `https://mojolang.org/docs/manual/<page>.md` before guessing an API. Full text: <https://mojolang.org/llms-manual.txt>.

## Do not generate these

| Stale | Write |
|---|---|
| `fn`, `let` | `def`, `var` |
| `class` | `struct` |
| `async` / `await` | Not in the language |
| `alias` | `comptime` |
| `@parameter if`, `@parameter for` | `comptime if`, `comptime for` |
| `read`, `borrowed` | Unmarked argument. Capture with `imm` |
| `inout`, `owned` | `mut`, `var` |
| `{read x}` | `{imm x}` |
| `var xs = [1, 2, 3]` treated as `List` | `var xs: List[Int] = [1, 2, 3]` |
| `s[i]`, `len(s)` on `String` | `s[byte=i]`, `s.byte_length()` |
| `f"..."` | `t"..."` |
| `from collections import` | `from std.collections import` |
| `ptr[i]` | `ptr[unsafe_offset=i]` |

`class`, `del`, `fn`, `global`, `nonlocal`, and `yield` are reserved and unused. `match` and `lambda` both exist.

## Variables

```mojo
var count = 0              # owned, mutable, type inferred
var name: String = "Mojo"
ref view = data[0]         # no copy; writes go through
comptime PI = 3.14159
```

`var` owns. `ref` views. Assigning an `ImplicitlyCopyable` value copies it. Transfer with `^`. A value is destroyed at its last use.

## Collections

A bare bracket literal is a fixed `Array`, not a `List`.

```mojo
var fixed = [1, 2, 3]                       # Array; length is part of the type
var nums: List[Int] = [1, 2, 3]             # growable
var scores: Dict[String, Int] = {"a": 1}
var flags: Set[Int] = {1, 2, 3}
var missing: Optional[Int] = None
print(missing.or_else(0))
```

- `List[Int](1, 2, 3)` is not a constructor.
- Empty `{}` is ambiguous. Annotate `Dict[K, V]` or `Set[T]`.
- An empty `Set` needs `from std.collections import Set`.
- `List` rejects negative indexes.
- `dict[key]` raises when the key is missing. `dict.get(key, default)` does not.

## Functions and ownership

An unmarked argument is an immutable borrow. Do not write `read`, `borrowed`, or `imm` on it.

| Convention | Meaning |
|---|---|
| unmarked | Immutable borrow. The default. |
| `mut` | Mutable borrow. Writes are visible to the caller. |
| `var` | Owns a copy, or the original when the caller passes `^`. |
| `ref` | Borrow whose mutability follows the origin. |
| `out` | Uninitialized slot the function must fill. No `->`. |
| `deinit` | Consumes the value. Destructors and consuming moves. |

```mojo
def add(mut x: Int, y: Int):
    x += y

def exclaim(var s: String):
    s += "!"

def make_point(out result: Point):
    result = Point(0, 0)

var g = "Hello"
exclaim(g)     # copy; g is still usable
exclaim(g^)    # transfer; g is uninitialized
```

`raises` is never implied: `def load() raises -> String`. A `def` without `raises` cannot raise. `Int`, `Float64`, and `SIMD` pass in registers.

## Closures and lambdas

Naming an outer value requires a capture list. A closure cannot outlive its scope.

| Capture | Meaning |
|---|---|
| `{imm x}` or `{x}` | Immutable borrow. The default. |
| `{mut x}` | Mutable borrow. |
| `{var x}` | Owned copy, taken when the closure is created. |
| `{var x^}` | Transfer the outer value in. |
| `{ref x}` | Mutability follows the origin. |
| `{}` | Captures nothing. |

```mojo
def scale(x: Int) {imm multiplier} -> Int:
    return x * multiplier

var inc = lambda (x: Int) -> Int: x + 1
bubble_sort(lambda (a: Int, b: Int) -> Bool: a > b, values)
```

A lambda body is one expression. A thin lambda (no captures) can be a compile-time function argument: `def(...) thin -> T`. A capturing lambda cannot.

## `match`

```mojo
match status:
    case 200:
        print("OK")
    case 301 | 302:
        print("Redirect")
    case code if code >= 500:
        print("Server", code)
    case _:
        print("other")

match reading:
    case .Some(temp) if temp > 20:
        return "warm"
    case .Some(_):
        return "cool"
    case .None:
        return "no reading"

if case .Some(temp) = reading:
    print(temp)
```

The first matching case runs. There is no `break` between cases. `_` matches anything and binds nothing. `comptime match` selects a case at compile time.

## Structs, traits, and lifecycle

```mojo
@fieldwise_init
struct Point(Copyable, Movable, Writable):
    var x: Int
    var y: Int

struct Handle(not Movable):
    pass

struct Lease(not Deinitable else "call release()"):
    def release(deinit self):
        pass
```

```mojo
def __init__(out self, x: Int):
    self.x = x

def __init__(out self, *, copy: Self):
    self.x = copy.x

def __init__(out self, *, deinit move: Self):
    self.x = move.x^

def __deinit__(deinit self):
    dealloc(self.allocation^)
```

`Self.T` names a parameter of the enclosing struct or trait. `self` is the instance. Behavior is reused through traits. Copy with `.copy()` unless the type is `ImplicitlyCopyable`. `return x^` transfers. There is no `-> T^`.

## Packages

A module is one `.mojo` file. A package is a directory that contains `__init__.mojo`. A file in another directory is importable only when that directory is a package.

```mojo
from mypackage.mymodule import MyPair
```

```bash
mojo precompile mypackage -o mypack.mojoc
```

A `.mojoc` loads only with the compiler version that built it. It is not an architecture-specific binary.

## Pointers

```mojo
from std.memory.alloc import alloc, dealloc, Layout

var allocation = alloc(Layout[Int](count=4))
var ptr = allocation.unsafe_ptr()
ptr.unsafe_offset(i).unsafe_write(i)   # uninitialized memory
print(ptr[unsafe_offset=i])            # initialized memory
dealloc(allocation^)
```

`ptr[]` reads or writes an initialized pointee. `unsafe_write` initializes. `unsafe_take_pointee` moves the value out. `alloc` does not return null; allocation failure aborts. Assigning through `ptr[]` on uninitialized memory is undefined.

## Numbers and strings

`Float32` is `Scalar[.float32]` is `SIMD[.float32, 1]`. SIMD widths are powers of two. Convert explicitly: `Float64(n)`, `v.cast[.float32]()`.

`/` truncates toward zero on integers and is true division on floats. `//` floors. `-7 / 2` is `-3`. `-7 // 2` is `-4`. The literal `7 / 2` is `3.5`.

`len(text)` does not compile. Use `byte_length()`, `count_codepoints()`, or `count_graphemes()`. Index with `text[byte=i]` or `text[codepoint=0:5]` (`StringSpan`). `'A'` is a one-character `String`. `t"{x:.2f}"` does not compile.

Stdlib imports use the `std.` prefix. Prelude names need no import: `Int`, `String`, `List`, `Dict`, `Optional`, `SIMD`, `print`, `range`.

## Errors and context managers

`raises` names the error type when it is not the default `Error`.

```mojo
def parse(s: String) raises Int -> Int:
    raise 42

def main() raises:
    try:
        _ = parse("bad")
    except err:          # err is Int
        print(err)
    finally:
        print("done")
```

A context manager implements `__enter__` and `__exit__`. `with` calls them. GPU puzzle files use `with DeviceContext() as ctx`. New code more often writes `var ctx = DeviceContext()`.

## Parameters

Parameters are compile-time and sit in `[]`. A `//` splits inferred parameters from parameters the caller writes.

```mojo
def scale[T: Copyable & Writable, //, n: Int](value: T) -> T:
    comptime assert n > 0, "n must be positive"
    return value

struct Box[T: AnyType]:
    var item: Self.T
```

`comptime assert` belongs in a function body, not at module scope. `Some[Writer]` is an existential. Conditional conformance uses `where conforms_to(T, Hashable)`.

## Traits

A trait lists required methods. The body `...` means the method has no default. A provided method is inherited by every conforming struct.

```mojo
trait Measurable:
    def length(self) -> Int: ...

struct Span(Measurable):
    def length(self) -> Int:
        return 0
```

There is no subclassing. Compose traits with `&`: `Copyable & Movable & Writable`.

Puzzle files use this same language. `fn`, `alias`, semicolons, and `from gpu import` are not current, including under `problems/`.
