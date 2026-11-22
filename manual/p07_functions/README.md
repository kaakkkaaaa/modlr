# Part 7: Advanced functions

Part 1 showed how to write a plain function. This part covers the rest: overloading, the full argument syntax, named results and returned references, and treating functions as values. It ends with closures and lambdas, the tools Mojo uses to pass behavior into algorithms, from a simple `map` to `vectorize`.

_Estimated time: 4–6 hours · Prerequisites: Parts 1–6_

Guide: [Part 7](https://claude.ai/artifact/76skpa3yrc1XGsrGXB4qf5#part-7)

## Goals

- Write overload sets and predict which overload a call picks.
- Use `/`, `*`, `*args`, `*args: *Ts`, and `var **kwargs` correctly.
- Return values through an `out` named result, and return references with `-> ref[...]`.
- Write closures with explicit capture lists and choose the right capture convention.
- Write lambdas, and know when a function can be a compile-time parameter (thin) and when it must be a runtime argument.
- Build generic higher-order functions that forward errors with parametric `raises`.

## Steps

| Done | Step | Topic | My file |
|:---:|---|---|---|
| [ ] | 7.1 | Overloaded functions and overload resolution | `s01_overloaded_functions_overload.mojo` |
| [ ] | 7.2 | Positional-only and keyword-only arguments | `s02_positional_only_keyword.mojo` |
| [ ] | 7.3 | Variadic arguments: `*args`, `*args: *Ts`, and `**kwargs` | `s03_variadic_arguments_args.mojo` |
| [ ] | 7.4 | Return values and named results with `out` | `s04_return_values_named.mojo` |
| [ ] | 7.5 | Returning references with `-> ref[origin]` | `s05_returning_references_ref.mojo` |
| [ ] | 7.6 | Functions as values and function types | `s06_functions_as_values.mojo` |
| [ ] | 7.7 | Closures and the capture list: `imm` and `mut` | `s07_closures_capture_list.mojo` |
| [ ] | 7.8 | Owning captures: `var`, `var x^`, and copyable closures | `s08_owning_captures_var.mojo` |
| [ ] | 7.9 | `ref` captures, empty capture lists, and mixing conventions | `s09_ref_captures_empty.mojo` |
| [ ] | 7.10 | Passing closures to higher-order functions | `s10_passing_closures_higher.mojo` |
| [ ] | 7.11 | Lambda expressions | `s11_lambda_expressions.mojo` |
| [ ] | 7.12 | Thin functions as compile-time parameters | `s12_thin_functions_as.mojo` |
| [ ] | 7.13 | Parametric `raises` for higher-order functions | `s13_parametric_raises_higher.mojo` |
| [ ] | 7.14 | Worked example: a small functional toolkit | `s14_worked_example_small.mojo` |

File names are suggestions. Keep them as valid identifiers (letters, digits, `_`, no leading digit) so they stay importable.

## Notes

<!-- What clicked, what confused me, links. One subsection per step works well. -->

## Checkpoint

Answer in your own words before opening the answers in the guide.

1. Two overloads differ only in their return type, `-> Int` and `-> Float64`. What happens, and why?

   > _My answer:_

2. Given `def f(a: Int, /, b: Int, *, c: Int = 0)`, which calls are valid: `f(1, 2)`, `f(a=1, b=2)`, `f(1, b=2, c=3)`, `f(1, 2, 3)`?

   > _My answer:_

3. Why must you iterate a heterogeneous `*args: *Ts` pack with `comptime for`, while `*args: Int` works with a normal `for`?

   > _My answer:_

4. When would you use an `out` named result instead of `-> T`?

   > _My answer:_

5. A closure is declared `def f() {imm a, var b, mut c}`. After it is defined, the outer code changes `a`, `b`, and `c`, and then calls `f()`. Which new values does `f` see? Which changes made by `f` are visible outside?

   > _My answer:_

6. Why does `inplace_transform[lambda (x: Int) -> Int: x * factor](xs)` fail when `factor` is a local variable, and what are two fixes?

   > _My answer:_

7. Exercise: write `def partition[T: Copyable & Deinitable, P: def(T) -> Bool, //](pred: P, xs: List[T]) -> Tuple[List[T], List[T]]` that returns the matching and non-matching items. Test it with a lambda that captures a threshold.

   > _My answer:_

8. What does `ErrorType` become in `run_action(get_value)` when `get_value` does not raise, and what does that buy the caller?

   > _My answer:_

## Manual pages

- [Functions](https://mojolang.org/docs/manual/functions/)
- [Closures](https://mojolang.org/docs/manual/functions/closures/)
- [Lambda expressions](https://mojolang.org/docs/manual/functions/lambda/)
- [Lifetimes, origins, and references](https://mojolang.org/docs/manual/values/lifetimes/)
- [Errors, error handling, and context managers](https://mojolang.org/docs/manual/errors/)
