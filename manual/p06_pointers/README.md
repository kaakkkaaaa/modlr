# Part 6: Pointers

Most Mojo code never touches a pointer: values, references, and argument conventions handle almost everything. This part covers the cases where you need one: a safe `Pointer(to=...)` reference you can store, the smart pointers `OwnedPointer` and `ArcPointer`, and manually allocated memory, where you take over jobs the compiler normally does for you. You finish by building a linked list that allocates and frees its own nodes.

_Estimated time: 4–6 hours · Prerequisites: Parts 1–5_

Guide: [Part 6](https://claude.ai/artifact/76skpa3yrc1XGsrGXB4qf5#part-6)

## Goals

- Pick the right pointer type (`Pointer`, `OwnedPointer`, `ArcPointer`) for a job, or decide you don't need one.
- Create safe pointers with `Pointer(to=x)` and read their origin and mutability in a signature.
- Run the full manual-memory lifecycle: `alloc`, `unsafe_write`, use, `unsafe_take_pointee`/`unsafe_deinit_pointee`, `dealloc`.
- Say exactly which operations are undefined behavior on uninitialized or freed memory.
- Store arrays of values, model null with `Optional[Pointer[...]]`, bitcast pointers, and do SIMD loads and stores.
- Write a self-referential data structure that cleans up after itself in `__deinit__`.

## Steps

| Done | Step | Topic | My file |
|:---:|---|---|---|
| [ ] | 6.1 | What a pointer is, and why you rarely need one | `s01_pointer_rarely_need.mojo` |
| [ ] | 6.2 | Vocabulary and the pointer-type map | `s02_vocabulary_pointer_type.mojo` |
| [ ] | 6.3 | Safe pointers: `Pointer(to=)` and origins | `s03_safe_pointers_pointer.mojo` |
| [ ] | 6.4 | `OwnedPointer`: one owner on the heap | `s04_ownedpointer_one_owner.mojo` |
| [ ] | 6.5 | `ArcPointer`: shared ownership with a reference count | `s05_arcpointer_shared_ownership.mojo` |
| [ ] | 6.6 | Manual memory: the five-stage lifecycle | `s06_manual_memory_five.mojo` |
| [ ] | 6.7 | Initialized vs uninitialized: the rules that prevent undefined behavior | `s07_initialized_uninitialized_rules.mojo` |
| [ ] | 6.8 | Storing multiple values | `s08_storing_multiple_values.mojo` |
| [ ] | 6.9 | More allocation patterns: `ThinAllocation`, errors, and `unsafe_leak` | `s09_allocation_patterns_thinallocation.mojo` |
| [ ] | 6.10 | Nullability with `Optional[Pointer[...]]` | `s10_nullability_optional_pointer.mojo` |
| [ ] | 6.11 | Foreign pointers, opaque pointers, and bitcasting | `s11_foreign_pointers_opaque.mojo` |
| [ ] | 6.12 | SIMD loads and stores through pointers | `s12_simd_loads_stores.mojo` |
| [ ] | 6.13 | Worked example: a self-referential linked list | `s13_worked_example_self.mojo` |
| [ ] | 6.14 | Safety checklist and old-code translation | `s14_safety_checklist_old.mojo` |

File names are suggestions. Keep them as valid identifiers (letters, digits, `_`, no leading digit) so they stay importable.

## Notes

<!-- What clicked, what confused me, links. One subsection per step works well. -->

## Checkpoint

Answer in your own words before opening the answers in the guide.

1. Why does `var b = a` work when `a` is a `Pointer` or an `ArcPointer`, but fail when it is an `OwnedPointer`? What happens to the pointee in each working case?

   > _My answer:_

2. After `var allocation = alloc(Layout[String](count=1))` and `var p = allocation.unsafe_ptr()`, what is wrong with `p[] = "hi"`? What should you write instead?

   > _My answer:_

3. Explain the difference between `unsafe_take_pointee()`, `unsafe_deinit_pointee()`, and `dealloc()`. Which of them run the value's destructor?

   > _My answer:_

4. Your function allocates, calls a raising function, then deallocates. The compiler says the allocation is "abandoned without being explicitly destroyed". Why, and what are two fixes?

   > _My answer:_

5. How do you represent "no next node" in a linked list, and why can't a struct contain an `Optional` of itself?

   > _My answer:_

6. What does `Pointer[mut=False, Byte, _]` in a parameter list mean, and can you pass a mutable pointer to it?

   > _My answer:_

7. Exercise: write a struct `FixedBuffer` that allocates `n` `Float64` values set to 0.0 in `__init__`, has `__getitem__`/`__setitem__` that raise on out-of-range indices, and frees its memory in `__deinit__`. Store a `ThinAllocation`.

   > _My answer:_

8. Exercise: in the linked list, why would calling `free_node(head)` inside `pop_front` (after `unsafe_take_pointee()`) be a bug?

   > _My answer:_

## Manual pages

- [Intro to pointers](https://mojolang.org/docs/manual/pointers/)
- [Using pointers](https://mojolang.org/docs/manual/pointers/using-pointers/)
- [Self-referential structs](https://mojolang.org/docs/manual/structs/reference/)
