# Part 4: Value lifecycle

Every Mojo value is born, may be copied or moved, and is destroyed exactly once. This part shows the methods that control each stage, when the compiler calls them, and how to use them to build types that own resources safely. Python hides all of this behind a garbage collector. Mojo puts it in your hands, with the compiler checking your work.

_Estimated time: 4–6 hours · Prerequisites: Parts 1–3_

Guide: [Part 4](https://claude.ai/artifact/76skpa3yrc1XGsrGXB4qf5#part-4)

## Goals

- Write initializers with `out self` and explain fieldwise versus logical initialization.
- Overload initializers, delegate between them, and enable implicit conversion with `@implicit` or braced initializer lists.
- Make a type copyable, move-only, or immovable, and write custom copy and move initializers when needed.
- Predict when Mojo destroys a value under its as-soon-as-possible (ASAP) policy, and extend a lifetime with `_ = x`.
- Write a `__deinit__()` that releases resources, and move values out of fields while tearing down.
- Design explicitly destroyed types with `not Deinitable` and use them with parameterized code.
- Build a heap-owning buffer type with correct allocation, copy, move, and destruction.

## Steps

| Done | Step | Topic | My file |
|:---:|---|---|---|
| [ ] | 4.1 | The lifecycle at a glance | `s01_lifecycle.mojo` |
| [ ] | 4.2 | Initializers, `out self`, and the two kinds of initialization | `s02_initializers_out_self.mojo` |
| [ ] | 4.3 | `@fieldwise_init`, overloads, and delegation | `s03_fieldwise_init_overloads.mojo` |
| [ ] | 4.4 | Implicit conversion and braced initializer lists | `s04_implicit_conversion_braced.mojo` |
| [ ] | 4.5 | Copying: `Copyable`, `.copy()`, and `ImplicitlyCopyable` | `s05_copying_copyable_copy.mojo` |
| [ ] | 4.6 | Moving: the `^` sigil and the move initializer | `s06_moving_sigil_move.mojo` |
| [ ] | 4.7 | Choosing movability: default, move-only, immovable | `s07_choosing_movability_default.mojo` |
| [ ] | 4.8 | Destruction: `__deinit__` and ASAP | `s08_destruction_deinit_asap.mojo` |
| [ ] | 4.9 | Inside a deinitializer: field order and moving out of fields | `s09_inside_deinitializer_field.mojo` |
| [ ] | 4.10 | Custom deinitializers for resources (RAII) | `s10_custom_deinitializers_resources.mojo` |
| [ ] | 4.11 | Explicit lifetime extension with `_ = x` | `s11_explicit_lifetime_extension.mojo` |
| [ ] | 4.12 | Explicit destruction: `not Deinitable` | `s12_explicit_destruction_not.mojo` |
| [ ] | 4.13 | Parameterized code and explicit destruction | `s13_parameterized_code_explicit.mojo` |
| [ ] | 4.14 | Worked example: a heap-owning buffer | `s14_worked_example_heap.mojo` |

File names are suggestions. Keep them as valid identifiers (letters, digits, `_`, no leading digit) so they stay importable.

## Notes

<!-- What clicked, what confused me, links. One subsection per step works well. -->

## Checkpoint

Answer in your own words before opening the answers in the guide.

1. What is the difference between fieldwise and logical initialization? Why does `var p: Person; p.name = "A"; p.age = 1; print(p)` fail?

   > _My answer:_

2. A struct has fields `name: String` and `tags: List[String]` and conforms to `Copyable`. Do you need to write a copy initializer? What about `var b = a`?

   > _My answer:_

3. Which of these run the move initializer: `var b = a^`; `take(a^)` where `take` has a `var` argument; `return T(1)`; `return t^` for a named local?

   > _My answer:_

4. Predict the destruction order for `print((a + b) * c)` using the `Number` type from step 4.8, with `a = Number(1)`, `b = Number(2)`, `c = Number(5)`.

   > _My answer:_

5. You create a lock guard with `var g = Guard("m")` and never mention `g` again. What goes wrong, and how do you fix it?

   > _My answer:_

6. Why can `__deinit__()` do `var x = self.field^` without refilling the field, while an ordinary `mut self` method cannot?

   > _My answer:_

7. When would you choose `not Deinitable` over a normal `__deinit__()`? Give two reasons.

   > _My answer:_

8. Exercise: write a move-only `FileId` type whose move initializer and deinitializer both print. Write `def reopen(var f: FileId) -> FileId` that returns its argument. How many moves and deinitializer calls do you see for `var g = reopen(f^)` followed by a `print` of `g`?

   > _My answer:_

## Manual pages

- [Value lifecycles](https://mojolang.org/docs/manual/lifecycle/)
- [Creating values](https://mojolang.org/docs/manual/lifecycle/life/)
- [Death of a value](https://mojolang.org/docs/manual/lifecycle/death/)
- [Explicit destruction](https://mojolang.org/docs/manual/lifecycle/explicit-destroy/)
- [Mojo structs](https://mojolang.org/docs/manual/structs/)
- [Parameterized declarations](https://mojolang.org/docs/manual/parameters/)
