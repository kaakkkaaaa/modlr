# Part 2: Language basics II: structs, traits, errors and packages

Part 1 gave you values, functions and control flow. This part gives you the tools to build your own types and organize real programs: structs with methods and operators, traits for code that works across many types, errors that are cheap enough to use everywhere, and the module, package and test layout every project needs.

_Estimated time: 6–9 hours · Prerequisites: Part 1_

Guide: [Part 2](https://claude.ai/artifact/76skpa3yrc1XGsrGXB4qf5#part-2)

## Goals

- Define structs with fields, initializers, methods, static methods and `comptime` members, and choose between `Copyable`, `ImplicitlyCopyable` and move-only.
- Explain how Mojo structs differ from Python classes.
- Make a type printable and give it operators: arithmetic, reversed, in-place, comparison, truthiness and subscripts.
- Define traits with required and default methods, conform to them, compose them with `&`, and write generic functions against them.
- Raise and handle errors with `try`/`except`/`else`/`finally`, define typed errors, and use `Never` and parametric raises.
- Use and write context managers.
- Split code into modules and packages, and write and run tests with `std.testing` and `TestSuite`.

## Steps

| Done | Step | Topic | My file |
|:---:|---|---|---|
| [ ] | 2.1 | Your first struct | `s01_first_struct.mojo` |
| [ ] | 2.2 | `@fieldwise_init`, traits in parentheses, and copying | `s02_fieldwise_init_traits.mojo` |
| [ ] | 2.3 | Methods, `mut self`, static methods and `comptime` members | `s03_methods_mut_self.mojo` |
| [ ] | 2.4 | Structs compared to Python classes | `s04_structs_compared_python.mojo` |
| [ ] | 2.5 | Special methods: printing and arithmetic operators | `s05_special_methods_printing.mojo` |
| [ ] | 2.6 | Comparisons, truthiness and subscripts | `s06_comparisons_truthiness_subscripts.mojo` |
| [ ] | 2.7 | Traits: contracts and generic functions | `s07_traits_contracts_generic.mojo` |
| [ ] | 2.8 | Default implementations and trait composition with `&` | `s08_default_implementations_trait.mojo` |
| [ ] | 2.9 | Raising and handling errors | `s09_raising_handling_errors.mojo` |
| [ ] | 2.10 | Typed errors and multiple error conditions | `s10_typed_errors_multiple.mojo` |
| [ ] | 2.11 | `Never`, parametric raises, and mixing error styles | `s11_never_parametric_raises.mojo` |
| [ ] | 2.12 | Context managers | `s12_context_managers.mojo` |
| [ ] | 2.13 | Modules and packages | `s13_modules_packages.mojo` |
| [ ] | 2.14 | Writing and running tests | `s14_writing_running_tests.mojo` |

File names are suggestions. Keep them as valid identifiers (letters, digits, `_`, no leading digit) so they stay importable.

## Notes

<!-- What clicked, what confused me, links. One subsection per step works well. -->

## Gotchas I hit

<!-- What I wrote, the error, the fix. -->

## Checkpoint

Answer in your own words before opening the answers in the guide.

1. A struct has `var count: Int = 0`. What happens, and how do you fix it?

   > _My answer:_

2. You declare `@fieldwise_init struct Box(Copyable): var items: List[Int]`. Which of these compile: `var b = a`, `var b = a.copy()`, `var b = a^`? Why not just make it `ImplicitlyCopyable`?

   > _My answer:_

3. Why does `def increment(self): self.count += 1` fail, and what is the fix?

   > _My answer:_

4. Your `Money` struct has `__add__(self, rhs: Float64)`. Why does `1.5 + m` fail, and what do you add? Which method handles `m += 1.5`?

   > _My answer:_

5. A struct defines `def area(self) -> Float64` but does not list the `Shape` trait that requires exactly that method. Can you pass it to `def f(s: Some[Shape])`?

   > _My answer:_

6. Write a function signature that accepts any type conforming to both `Writable` and `Equatable`, two arguments of the same type, and returns whether they are equal while printing both.

   > _My answer:_

7. A function must report three distinct failure causes and callers must branch on the cause. Which error design do you use, and how does the caller branch?

   > _My answer:_

8. What does each of these mean: `raises -> Never`, `raises Never -> Int`, and a bare `raises` on a function that only calls a `raises ValidationError` function?

   > _My answer:_

9. You moved your structs into `lib/shapes.mojo` and `main.mojo` cannot import them. List what is needed. Then write a test file skeleton for them.

   > _My answer:_

## Manual pages

- [Mojo structs](https://mojolang.org/docs/manual/structs/)
- [Add operator support to custom types](https://mojolang.org/docs/manual/structs/operator-support/)
- [Traits](https://mojolang.org/docs/manual/traits/)
- [Errors, error handling, and context managers](https://mojolang.org/docs/manual/errors/)
- [Modules and packages](https://mojolang.org/docs/manual/packages/)
- [Testing](https://mojolang.org/docs/tools/testing/)
