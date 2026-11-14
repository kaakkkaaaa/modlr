---
name: mojo-language-reference
---

# Mojo language reference

Docs 1.2.0 override `mojo-syntax`. Fetch the page when a form is not listed here.

Source: <https://mojolang.org/docs/reference/>. Full text: <https://mojolang.org/llms-reference.txt>.

## Keywords

```text
Control       if elif else for while break continue pass return match case with
Errors        try except finally raise assert
Declarations  def lambda struct trait var ref
Operators     and or not in is
Imports       import from as
Compile-time  comptime  type_of  origin_of  conforms_to
Literals      True False None Self
```

Reserved and unused: `class`, `del`, `fn`, `global`, `nonlocal`, `yield`.

`True`, `None`, and `Self` are case-sensitive. `self` is a conventional argument name, not a keyword. Escape a keyword with backticks: `` `struct` ``.

`mut`, `out`, `deinit`, and `imm` are conventions, not reserved words. They cannot be argument names. `var` and `ref` are keywords.

## Argument and capture conventions

```mojo
def f(x: Int): ...            # immutable borrow, unmarked
def g(mut x: Int): ...
def h(var x: String): ...     # h(s) copies, h(s^) transfers
def make(out result: Point): ...
def __deinit__(deinit self): ...
def view(ref self) -> ref[self] Self.T: ...

def closure(x: Int) {imm limit, mut total, var snapshot} -> Int: ...
var inc = lambda (x: Int) {imm c} -> Int: x + c
```

| Marker | Meaning |
|---|---|
| `//` | Infer-only parameters. Callers omit them. |
| `/` | Positional-only. |
| `*` | Keyword-only. |

`*args: T` is a same-type variadic. `*args: *Ts` is a heterogeneous pack. A function type is `def(Int, Int) -> Int`, with no argument names. `thin` means the function has no captures.

`out` cannot be combined with `->`. On a closure, `raises` sits before the capture list.

## Operators

| Operators | Meaning |
|---|---|
| `+` `-` `*` `/` `//` `%` `**` | `**` is power. `^` is not power. |
| `==` `!=` `<` `<=` `>` `>=` | Comparisons chain: `a < b <= c`. |
| `and` `or` `not` | Short-circuit. |
| `&` `\|` `^` `~` `<<` `>>` | Bitwise. `^` is XOR. |
| `value^` | Ownership transfer. |

`/` truncates toward zero on integers. `//` floors. Numeric conversions are explicit.

## Literals and types

```mojo
42  0xFF  0o52  0b1010  1_000_000      # Int at runtime
3.14  .5  2.  2.5e-3                   # Float64 at runtime
SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0) # leading-dot DType
"hi"  r"C:\raw"  t"x={n}"  rt"\path {n}"
```

- `t"..."` has no format specifiers. `t"{x:.2f}"` is an error. `{{` is a literal brace.
- Adjacent string literals concatenate. `'A'` is a `String`, not a character.
- `2 ** 200` stays an arbitrary-precision `IntLiteral` at compile time.
- Leading zeros on base-10 integers are rejected.
- `len(s)` on `String` does not compile. Use `byte_length`, `count_codepoints`, or `count_graphemes`.
- `s[byte=i]` and `s[codepoint=start:end]` produce a `StringSpan`.
- `[1, 2, 3]` is `Array`. `{"k": v}` is `Dict`. `{1, 2}` is `Set`. `(1, "a")` is `Tuple`.

## Decorators

```mojo
@inline(.always)      # also .nodebug, .never, .automatic
@fieldwise_init
@implicit             # initializer may be used as a conversion
@staticmethod
@deprecated("msg")
@doc_hidden
@export               # export from a shared library
@align(N)             # minimum alignment of a struct
```

`@always_inline` and `@no_inline` are legacy. `@__parameter` and `@__copy_capture` are deprecated. Prefer `@inline`.

## `match` patterns

```mojo
match value:
    case 1 | 2:          # alternatives
        pass
    case (x, y):         # tuple; names bind
        pass
    case .Some(n) if n > 0:
        pass
    case _:              # matches anything, binds nothing
        pass
```

Cases do not fall through. `if case pattern = value:` tests one pattern. `comptime match` picks a case at compile time. Patterns also open tuples, struct fields, and list elements.

## Functions, structs, traits

```mojo
def name[P: Trait, //, n: Int](mut x: T, *items: Int) raises -> Int where n > 0:
    ...

def make(out result: Point):
    result = Point(0, 0)

struct Pair[T: Copyable](Equatable):
    var a: Self.T
    var b: Self.T

    def __init__(out self, a: Self.T, b: Self.T):
        self.a = a
        self.b = b

trait Showable:
    def write_to(self, mut writer: Some[Writer]): ...
```

`raises` sits before `->` and before a closure's `{captures}`. `where` is the constraint clause. `...` as a function body marks a required trait method. An `@staticmethod` has no `self`.

Copy and move constructors are keyword-only: `__init__(out self, *, copy: Self)` and `__init__(out self, *, deinit move: Self)`. The destructor is `__deinit__(deinit self)`.

`not Movable` pins a value. `not Deinitable else "reason"` makes it linear. The compiler prints `reason` when a path drops it.
