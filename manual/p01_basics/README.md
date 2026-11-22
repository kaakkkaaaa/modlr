# Part 1: Language basics

This part teaches the everyday core of Mojo: program structure, variables, the built-in types, operators, control flow, functions, and the basics of raising errors. Mojo looks like Python, but it is statically typed and compiled, so many small Python habits produce compile errors. Learning those differences now saves hours later.

_Estimated time: 5–7 hours · Prerequisites: none (Mojo 1.2.0 installed and `mojo` on your PATH)_

Guide: [Part 1](https://claude.ai/artifact/76skpa3yrc1XGsrGXB4qf5#part-1)

## Goals

- Write, run, and build a Mojo program with a `main()` entry point.
- Declare typed variables with `var` and constants with `comptime`, and explain block scoping and shadowing.
- Choose between `Int`, sized integers, and float types, convert between them explicitly, and use basic `SIMD` vectors.
- Work with UTF-8 strings using byte, codepoint, and grapheme views, and format text with t-strings.
- Use `Array`, `List`, `Dict`, `Set`, `Optional`, and tuples, and know when each is right.
- Predict the result of integer division, floor division, and modulo for negative numbers.
- Write `if`, `while`, `for`, `match`, `if case`, and `while case` statements.
- Define functions with default, keyword, positional-only, keyword-only, and variadic arguments, and mark raising functions with `raises`.

## Steps

All 14 steps are written. Run a file from `manual/` with `pixi run mojo run p01_basics/s01_hello.mojo`.

| Done | Step | Topic | File |
|:---:|---|---|---|
| [✓] | 1.1 | Hello world and program structure | `s01_hello.mojo` |
| [✓] | 1.2 | Variables, static types, and scope | `s02_variables.mojo` |
| [✓] | 1.3 | Numeric types and conversions | `s03_numbers.mojo` |
| [✓] | 1.4 | SIMD and DType: what numbers really are | `s04_simd.mojo` |
| [✓] | 1.5 | Strings: UTF-8, indexing, and t-strings | `s05_strings.mojo` |
| [✓] | 1.6 | Booleans, truthiness, and tuples | `s06_bools_tuples.mojo` |
| [✓] | 1.7 | Array vs List | `s07_array_list.mojo` |
| [✓] | 1.8 | Dict, Set, and Optional | `s08_dict_set_optional.mojo` |
| [✓] | 1.9 | Operators | `s09_operators.mojo` |
| [✓] | 1.10 | Control flow: if, while, for | `s10_control_flow.mojo` |
| [✓] | 1.11 | Pattern matching: match, if case, while case | `s11_match.mojo` |
| [✓] | 1.12 | Functions and arguments | `s12_functions.mojo` |
| [✓] | 1.13 | Raising and non-raising functions | `s13_raises.mojo` |
| [✓] | 1.14 | Putting it together, and Python habits to unlearn | `s14_wrap_up.mojo` |

## Notes

<!-- What clicked, what confused me, links. One subsection per step works well. -->

## Checkpoint

Answer in your own words before opening the answers in the guide.

1. What does `var xs = [4, 5, 6]` create, and why does `xs.append(7)` fail?

   > _My answer:_

2. Given `var a = -9` and `var b = 4`, what do `a / b`, `a // b`, and `a % b` print? What does the literal expression `9 / 4` print?

   > _My answer:_

3. Why does this fail, and how do you fix it? `var n: Int = 3` then `var avg: Float64 = 10.0 / n`.

   > _My answer:_

4. For `var s = String("naïve")`, how do you get the number of characters, the third character, and the number of bytes?

   > _My answer:_

5. This function does not compile. Give two different fixes. def lookup(d: Dict[String, Int], key: String) -> Int: return d[key]

   > _My answer:_

6. Rewrite this using `if case`: `if opt: print(opt.value())` where `opt` is an `Optional[Int]`. Why is the new form better?

   > _My answer:_

7. Which calls are valid for `def area(w: Int, h: Int, /, *, double: Bool = False)`? (a) `area(2, 3)` (b) `area(w=2, h=3)` (c) `area(2, 3, True)` (d) `area(2, 3, double=True)`

   > _My answer:_

8. Exercise: write a program with a function `stats(values: List[Int]) raises -> Tuple[Int, Int, Float64]` that returns the minimum, the maximum, and the mean, and raises on an empty list. In `main()`, call it on a non-empty list and an empty list, unpack the result, and handle the error.

   > _My answer:_

## Manual pages

- [Mojo language basics](https://mojolang.org/docs/manual/basics/)
- [Mojo quickstart](https://mojolang.org/docs/manual/quickstart/)
- [Variables](https://mojolang.org/docs/manual/variables/)
- [Types](https://mojolang.org/docs/manual/types/)
- [Operators](https://mojolang.org/docs/manual/operators/)
- [Control flow](https://mojolang.org/docs/manual/control-flow/)
- [Functions](https://mojolang.org/docs/manual/functions/)
- [Mojo tips for Python devs](https://mojolang.org/docs/manual/python-to-mojo/)
