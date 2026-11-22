# Part 3: Value ownership

Mojo has no garbage collector and no manual `free()`. Every value has exactly one owner, the compiler tracks who that is, and the value is destroyed right after its last use. This part teaches the rules, the argument conventions that express them, and the references and origins that let you share data without copying it.

_Estimated time: 4–6 hours · Prerequisites: Parts 1–2_

Guide: [Part 3](https://claude.ai/artifact/76skpa3yrc1XGsrGXB4qf5#part-3)

## Goals

- Explain where values live (stack or heap) and how ownership differs from garbage collection and manual memory management.
- Predict when Mojo copies, when it refuses to copy, and when you must write `.copy()` or `^`.
- Choose the right argument convention: default (immutable), `mut`, `var`, `ref`, `out`, or `deinit`.
- Explain argument exclusivity and why `f(x, x)` can be a compile error.
- Use the `^` transfer sigil, and know what state it leaves the source variable in.
- Predict when a value is destroyed under ASAP destruction.
- Create reference bindings with `ref`, and write functions that take and return references with origins.
- Read the most common ownership error messages and fix them.

## Steps

| Done | Step | Topic | My file |
|:---:|---|---|---|
| [ ] | 3.1 | Why ownership: stack, heap, and who frees memory | `s01_ownership_stack_heap.mojo` |
| [ ] | 3.2 | Value semantics: copies are independent | `s02_value_semantics_copies.mojo` |
| [ ] | 3.3 | Copy or transfer: `ImplicitlyCopyable`, `.copy()`, and `^` | `s03_copy_transfer_implicitlycopyable.mojo` |
| [ ] | 3.4 | Argument conventions at a glance | `s04_argument_conventions.mojo` |
| [ ] | 3.5 | The default: immutable references | `s05_default_immutable_references.mojo` |
| [ ] | 3.6 | `mut`: changing the caller's value | `s06_mut_changing_caller.mojo` |
| [ ] | 3.7 | Argument exclusivity: no aliasing of `mut` | `s07_argument_exclusivity_no.mojo` |
| [ ] | 3.8 | `var` arguments and the `^` transfer sigil | `s08_var_arguments_transfer.mojo` |
| [ ] | 3.9 | Destruction at last use (ASAP) | `s09_destruction_last_use.mojo` |
| [ ] | 3.10 | `out` and `deinit`: initializing and consuming | `s10_out_deinit_initializing.mojo` |
| [ ] | 3.11 | Reference bindings with `ref` | `s11_reference_bindings_ref.mojo` |
| [ ] | 3.12 | Origins: how the compiler tracks references | `s12_origins_compiler_tracks.mojo` |
| [ ] | 3.13 | `ref` arguments and `ref` return values | `s13_ref_arguments_ref.mojo` |
| [ ] | 3.14 | Common ownership errors and how to read them | `s14_common_ownership_errors.mojo` |

File names are suggestions. Keep them as valid identifiers (letters, digits, `_`, no leading digit) so they stay importable.

## Notes

<!-- What clicked, what confused me, links. One subsection per step works well. -->

## Checkpoint

Answer in your own words before opening the answers in the guide.

1. What does `var b = a` do when `a` is an `Int`, a `String`, and a `List[Int]`?

   > _My answer:_

2. A function is declared `def f(var s: String)`. List the three ways a caller can provide `s`, and what happens to the caller's variable in each.

   > _My answer:_

3. Why does `append_twice(my_string, my_string)` fail when `append_twice(n, n)` with `Int` arguments works?

   > _My answer:_

4. In the program below, when is `a` destroyed? `var a = Noisy("a"); use(a); print("x"); print("y")`

   > _My answer:_

5. Fix this function so it compiles and the caller's list is not changed: `def plus_one(l: List[Int]) -> List[Int]: l.append(1); return l`

   > _My answer:_

6. What is the difference between `var x = items[0]` and `ref x = items[0]`?

   > _My answer:_

7. Write `def last(ref list: List[String]) -> ...` that returns a reference to the last element, so that `last(names) += "!"` works.

   > _My answer:_

8. Why does `pick_one()` return `ref[a, b] String` rather than `ref[a] String`?

   > _My answer:_

## Manual pages

- [Intro to value ownership](https://mojolang.org/docs/manual/values/)
- [Value semantics](https://mojolang.org/docs/manual/values/value-semantics/)
- [Ownership](https://mojolang.org/docs/manual/values/ownership/)
- [Lifetimes, origins, and references](https://mojolang.org/docs/manual/values/lifetimes/)
- [Variables](https://mojolang.org/docs/manual/variables/)
- [Death of a value](https://mojolang.org/docs/manual/lifecycle/death/)
