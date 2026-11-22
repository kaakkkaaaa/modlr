# Part 5: Metaprogramming

Mojo runs ordinary Mojo code at compile time. You use the same language to compute constants, unroll loops, pick code paths, and generate specialized versions of functions and structs. This part teaches parameters, `comptime`, constraints, and reflection. These are the tools the standard library uses to give you generic code with no run-time cost.

_Estimated time: 6–8 hours · Prerequisites: Parts 1–4_

Guide: [Part 5](https://claude.ai/artifact/76skpa3yrc1XGsrGXB4qf5#part-5)

## Goals

- Tell parameters (`[]`, compile time) from arguments (`()`, run time) and choose the right one.
- Use `comptime` constants, `comptime if`, `comptime for`, and `comptime (...)` sub-expressions.
- Write parameterized functions and structs with type and value parameters, defaults, keyword parameters, infer-only parameters (`//`), and variadic parameters.
- Bound type parameters with traits, use `Some[Trait]`, and add conditional conformance with `where`.
- Write constraints and `comptime assert`, and fix "lacking evidence" errors.
- Control materialization of compile-time values and build static lookup tables.
- Use unbound and partially-bound types, automatic parameterization, `type_of`, and `rebind`.
- Inspect structs with `reflect[T]`, and switch features with `-D` defines and platform checks.

## Steps

| Done | Step | Topic | My file |
|:---:|---|---|---|
| [ ] | 5.1 | Parameters vs arguments | `s01_parameters_arguments.mojo` |
| [ ] | 5.2 | `comptime` constants and compile-time evaluation | `s02_comptime_constants_compile.mojo` |
| [ ] | 5.3 | `comptime if` and `comptime for` | `s03_comptime_if_comptime.mojo` |
| [ ] | 5.4 | Parameterized functions: type and value parameters | `s04_parameterized_functions_type.mojo` |
| [ ] | 5.5 | Parameterized structs, `Self.T`, and `comptime` members | `s05_parameterized_structs_self.mojo` |
| [ ] | 5.6 | Parameter declarations: defaults, keywords, `//`, variadics | `s06_parameter_declarations_defaults.mojo` |
| [ ] | 5.7 | Traits as bounds, `Some[Trait]`, and safe downcasting | `s07_traits_as_bounds.mojo` |
| [ ] | 5.8 | Conditional conformance and `where` on methods | `s08_conditional_conformance_where.mojo` |
| [ ] | 5.9 | Constraints and `comptime assert` | `s09_constraints_comptime_assert.mojo` |
| [ ] | 5.10 | Materialization and lookup tables | `s10_materialization_lookup_tables.mojo` |
| [ ] | 5.11 | Parameterized `comptime`, unbound types, and automatic parameterization | `s11_parameterized_comptime_unbound.mojo` |
| [ ] | 5.12 | Reflection | `s12_reflection.mojo` |
| [ ] | 5.13 | Compile-time defines, platform checks, and `debug_assert` | `s13_compile_time_defines.mojo` |
| [ ] | 5.14 | Worked example: a SIMD vector type | `s14_worked_example_simd.mojo` |

File names are suggestions. Keep them as valid identifiers (letters, digits, `_`, no leading digit) so they stay importable.

## Notes

<!-- What clicked, what confused me, links. One subsection per step works well. -->

## Checkpoint

Answer in your own words before opening the answers in the guide.

1. What is the difference between `f[3]()` and `f(3)`? Why can't you write `f[n]()` when `n` is a `var`?

   > _My answer:_

2. You have `def f[dtype: DType, value: Scalar[dtype]]()`. Callers complain they must write `f[DType.float32, Float32(1)]()`. How do you let them write `f[Float32(1)]()`?

   > _My answer:_

3. This code fails with "cannot materialize comptime value of type 'List[Int]'". Give two fixes. `comptime xs: List[Int] = [1, 2, 3]` then `for i in range(n): total += xs[i % 3]`.

   > _My answer:_

4. Why does this fail, and what are the three ways to fix it? `def g[n: Int]() where n > 5: h[n]()` where `h` requires `where n > 3`.

   > _My answer:_

5. Write a struct `Box[T: Copyable & Deinitable]` that is `Writable` only when `T` is, and has a method `doubled()` available only when `T` is `Int`-like.

   > _My answer:_

6. What does `def f(v: SIMD[DType.float32, _])` mean, and how do you read the width inside?

   > _My answer:_

7. When is `rebind` the right tool, and when is it wrong?

   > _My answer:_

8. Exercise: write `def field_report[T: AnyType]()` that prints each field name and whether it is `Writable`, using `reflect`, `comptime for`, and `conforms_to`.

   > _My answer:_

## Manual pages

- [Intro to metaprogramming](https://mojolang.org/docs/manual/metaprogramming/)
- [Compile-time evaluation](https://mojolang.org/docs/manual/metaprogramming/comptime-evaluation/)
- [Parameterization](https://mojolang.org/docs/manual/parameters/)
- [Parameterized declarations](https://mojolang.org/docs/manual/generics/)
- [Materializing compile-time values](https://mojolang.org/docs/manual/metaprogramming/materialization/)
- [Reflection](https://mojolang.org/docs/manual/metaprogramming/reflection/)
