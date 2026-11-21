# modlr

<p align="center">
  <img src="https://miro.medium.com/0*wS9xIg1uRHIbP4OP.jpeg" alt="Modular, the next-generation AI developer platform" width="690">
</p>

My notes, exercises, and solutions while learning the [Mojo](https://mojolang.org/) programming language, split into four independent tracks.

| Track | Source | Mojo | Status |
|---|---|---|---|
| [problems/](#gpu-puzzles) | [Mojo GPU Puzzles](https://puzzles.modular.com/) | 1.2.0 for p01–p06; 0.25.7 nightly for p07–p15 | p01–p05 kernels on 1.2.0. p15 not started |
| [advent-of-code/](#advent-of-mojo) | [Advent of Mojo](https://advent.mojolang.org/) | 1.2.0 | Chapters 1–8 |
| [manual/](#mojo-manual-guides) | [Mojo manual](https://mojolang.org/docs/manual/) | 1.2.0 | Part 1 through step 1.12 |
| [blog/](#blog) | Posts updated for Mojo 1.2.0 | 1.2.0 | SIMD matrix multiply, parallel axis sums |

```
.
├── problems/         # one Pixi env per puzzle
├── advent-of-code/   # one shared Pixi env
├── manual/           # one shared Pixi env
└── blog/             # one folder per post
```

## Before you start

- **Install [Pixi](https://pixi.sh/).** `problems/`, `advent-of-code/`, and `manual/` each manage Mojo through Pixi. There is no environment at the repository root, and those tracks do not share a Mojo version, so always `cd` into a track (or puzzle) folder first. `blog/` is the exception: those programs use the Mojo 1.2.0 compiler already on your `PATH`.
- **Apple Silicon only.** Every `pixi.toml` targets `osx-arm64`. On another platform, add your platform to `platforms` and run `pixi install` to regenerate the lock file.

## GPU Puzzles

Solutions to [Mojo GPU Puzzles](https://puzzles.modular.com/). The sequence starts with one thread writing one output element, then adds bounds checks, 2D indexing, blocks, shared memory, and full kernels: pooling, dot product, convolution, prefix sum, and matrix multiplication.

Each `problems/pXX/` directory is its own Pixi project (`pixi.toml` and `pixi.lock`). A kernel file is a GPU function plus a `main` that allocates device buffers with `DeviceContext`, launches the kernel, and checks the result with `assert_equal`.

### Version

> [!NOTE]
> p01–p06 target **Mojo 1.2.0**. p07–p15 still target **Mojo 0.25.7** and will be updated to 1.2.0, the same version already used by `manual/`, `advent-of-code/`, and `blog/`.

The first six puzzles have moved off the nightly channel. Their `pixi.toml` files point at the [stable MAX channel](https://conda.modular.com/max) and pin both the compiler and the GPU package. `mojo` by itself does not ship `max.gpu`, so a 1.2.0 environment without `max` cannot compile these kernels.

| Folders | Channel | `pixi.toml` | What that means |
|---|---|---|---|
| p01–p06 | [stable MAX](https://conda.modular.com/max) | `mojo = "1.2.*"`, `max = "26.7.*"` | The lock resolves to Mojo 1.2.0 and MAX 26.7.0 |
| p07–p15 | [MAX nightly](https://conda.modular.com/max-nightly) | `mojo = ">=0.25.7.0.dev2025100505,<0.26"` | The older nightly these folders were first locked to |

What already landed in p01–p05:

1. The environment above, with a fresh `pixi.lock` for `osx-arm64`.
2. The 1.2.0 GPU API. Imports come from `max.gpu` and `max.gpu.host`. Kernels are `def`, compile-time values are `comptime`, and device pointers are `Pointer[..., MutAnyOrigin]` indexed with `unsafe_offset`. Launches use `enqueue_function`, and scalar sizes passed into a kernel are `Int32`.
3. `TileTensor` in place of the old `LayoutTensor` files. p04 keeps a pointer kernel (`p04.mojo`) next to `p04-tile-tensor-intro.mojo` and `p04-tile-tensor.mojo`. p05 is only the broadcast kernel, and it indexes the operands as `a[0, col] + b[row, 0]`.
4. A run of each of those files with `pixi run mojo`. The existing `assert_equal` checks passed.

`p06.mojo` is the exception inside that group. Its Pixi environment is already Mojo 1.2.0, but the kernel is still the 0.25.7 version: `alias`, `fn`, `UnsafePointer`, and `from gpu import ...`. It will not compile in the 1.2.0 environment until it is ported the same way as p01–p05.

The rest of the update, for p07–p15, will do three things:

1. Point each folder at the stable MAX channel and pin `mojo = "1.2.*"` and `max = "26.7.*"`.
2. Bring the kernels in line with 1.2.0. Those files still use `alias`, `fn`, `UnsafePointer` without an origin, and, where a puzzle has a second version, `LayoutTensor`.
3. Re-lock each `pixi.lock` on `osx-arm64` after the code compiles and the existing `assert_equal` checks pass.

Until that lands, install and run a puzzle with the Pixi environment inside its own folder. A p01 environment will not compile a p08 kernel, and the other way around.

### Run

`pixi install` is once per puzzle. After that, `pixi run mojo` uses the Mojo binary from that folder's lock file.

```bash
cd problems/p01
pixi install
pixi run mojo p01.mojo
```

For a puzzle with several programs, pass the file you want. From `problems/p14`:

```bash
pixi run mojo p14-tiled-matmul.mojo
```

### Puzzles

| Puzzle | Topic | Files |
|---|---|---|
| [p01](problems/p01/) | Map: add 10 on one thread per element | `p01.mojo` |
| [p02](problems/p02/) | Zip: add two vectors | `p02.mojo` |
| [p03](problems/p03/) | Guard: skip threads past the array length | `p03.mojo` |
| [p04](problems/p04/) | 2D map. Pointer kernel, then the same add with `TileTensor` | `p04.mojo`, `p04-tile-tensor-intro.mojo`, `p04-tile-tensor.mojo` |
| [p05](problems/p05/) | Broadcast add. One `TileTensor` kernel, row vector plus column vector | `p05.mojo` |
| [p06](problems/p06/) | One block grid. Environment is 1.2.0; the kernel is still 0.25.7 | `p06.mojo` |
| [p07](problems/p07/) | 2D block grid | `p07.mojo`, `p07-layout-tensor.mojo` |
| [p08](problems/p08/) | Shared memory | `p08.mojo`, `p08-layout-tensor.mojo` |
| [p09](problems/p09/) | Pooling, including a 2D kernel | `p09.mojo`, `p09-layout-tensor.mojo`, `p09-pooling-2d-pt*.mojo` |
| [p10](problems/p10/) | Dot product | `p10.mojo`, `p10-layout-tensor.mojo` |
| [p11](problems/p11/) | 1D and 2D convolution, including block boundaries | `p11.mojo`, `p11-conv*.mojo`, `p11-block-boundary.mojo` |
| [p12](problems/p12/) | Prefix sum: local phase and across block boundaries | `p12.mojo`, `p12-block-boundary.mojo` |
| [p13](problems/p13/) | Axis sums: basic, shared memory, and warp-level | `p13-axis-*.mojo` |
| [p14](problems/p14/) | Matmul: naive, tiled, coalesced, register-tiled, SIMD | `p14.mojo`, `p14-*-matmul*.mojo` |
| [p15](problems/p15/) | Not started (`p15.mojo` is empty) | `p15.mojo` |

Several kernels still have the original `# FILL ME IN` comment sitting above a filled-in body. The comment marks the line the puzzle asked for. The code under it is the current attempt, and `main` checks that attempt with `assert_equal`.

## Advent of Mojo

Guided notes for [Advent of Mojo](https://advent.mojolang.org/), a short course that goes from installing Mojo to reading a sensor log and answering questions about it. One shared environment covers every chapter: `advent-of-code/pixi.toml` with `mojo = ">=1.2.0,<2"` on the stable MAX channel. Each chapter README is the walkthrough: the idea, the code, and the expected output. Sample inputs live in that chapter's `src/downloads/`.

| Chapter | Program | What it covers |
|---|---|---|
| [p01 - Cover](<advent-of-code/p01 - Cover/>) | Cover art only | Light and dark cover images |
| [p02 - Welcome to Advent Mojo](<advent-of-code/p02 - Welcome to Advent Mojo/>) | Install notes | System requirements, Pixi, and the Modular channels |
| [p03 - Mojo basics](<advent-of-code/p03 - Mojo basics/>) | `analyzer.mojo` | Functions, lists, averages, and raising on an empty list |
| [p04 - Work with files](<advent-of-code/p04 - Work with files/>) | `temp_log.mojo` | Read a log, write a report, append a line, then delete the report |
| [p05 - Work with strings](<advent-of-code/p05 - Work with strings/>) | `string_tour.mojo` | Length, slices, search, trim, reverse, and `split` |
| [p06 - Parse text into typed data](<advent-of-code/p06 - Parse text into typed data/>) | `parse_text.mojo` | Split lines into fields and parse temperatures as `Float64` |
| [p07 - Pull numbers from text](<advent-of-code/p07 - Pull numbers from text/>) | `pull_numbers_from_text.mojo` | Scan digit runs, keep the day numbers, parse one signed decimal |
| [p08 - Count and look up values](<advent-of-code/p08 - Count and look up values/>) | `count_and_look_up_values.mojo` | Count with `Dict`, test membership with `Set`, key by `(station, day)` |

Install the shared environment once from `advent-of-code`. Chapters p04–p08 open `src/downloads/` relative to the current directory, so run each of those programs from its own chapter folder:

```bash
cd advent-of-code
pixi install

cd "p08 - Count and look up values"
pixi run --manifest-path ../pixi.toml mojo count_and_look_up_values.mojo
```

p08 downloads `station_reports.txt` into `src/downloads/` on its first run, so it needs network access.

## Mojo Manual Guides

Study notes for the Mojo **1.2.0** manual, one file per step. The environment (`manual/pixi.toml`) pins `mojo = "1.2.*"` and adds Python 3.12 and NumPy for Part 8, where Mojo calls Python. [manual/README.md](manual/README.md) is the progress table for all eight parts. Each part folder has its own README with the step checklist, goals, and checkpoint questions.

Part 1 is the only part with programs so far: `s01_hello.mojo` through `s12_functions.mojo`. Those files are small programs you can run, covering `main`, `var` and `comptime`, numbers, SIMD, strings, tuples, `Array` and `List`, `Dict`, `Set`, `Optional`, operators, loops, `match`, and function arguments. Steps 1.13–1.14 (raising, and a closing exercise) are still unwritten. Parts 2–8 are checklists waiting for the same treatment: structs and traits, ownership, lifecycle, metaprogramming, pointers, advanced functions, and Python/C interop.

```bash
cd manual
pixi install
pixi run version                                              # prints the Mojo version
pixi run mojo run p01_basics/s01_hello.mojo                   # run a file
pixi run mojo build p01_basics/s01_hello.mojo -o build/hello  # build a binary
```

| Part | Topic | Status |
|---|---|---|
| [p01_basics](manual/p01_basics/) | Language basics | `s01`–`s12` written. Steps 1.13–1.14 to do |
| [p02_structs_traits](manual/p02_structs_traits/) | Structs, traits, errors, packages | Checklist only |
| [p03_ownership](manual/p03_ownership/) | Value ownership | Checklist only |
| [p04_lifecycle](manual/p04_lifecycle/) | Value lifecycle | Checklist only |
| [p05_metaprogramming](manual/p05_metaprogramming/) | Metaprogramming | Checklist only |
| [p06_pointers](manual/p06_pointers/) | Pointers | Checklist only |
| [p07_functions](manual/p07_functions/) | Advanced functions | Checklist only |
| [p08_interop](manual/p08_interop/) | Python and C interop | Checklist only |

<details>
<summary>Part 1 files</summary>

| File | Step |
|---|---|
| `s01_hello.mojo` | Hello world and program structure |
| `s02_variables.mojo` | Variables, static types, and scope |
| `s03_numbers.mojo` | Numeric types and conversions |
| `s04_simd.mojo` | SIMD and `DType` |
| `s05_strings.mojo` | Strings, indexing, and t-strings |
| `s06_bools_tuples.mojo` | Booleans, truthiness, and tuples |
| `s07_array_list.mojo` | `Array` and `List` |
| `s08_dict_set_optional.mojo` | `Dict`, `Set`, and `Optional` |
| `s09_operators.mojo` | Operators |
| `s10_control_flow.mojo` | `if`, `while`, and `for` |
| `s11_match.mojo` | `match`, `if case`, and `while case` |
| `s12_functions.mojo` | Functions and arguments |

</details>

## Blog

Write-ups that sit next to runnable programs. There is no Pixi project in this track. Both posts were rewritten in November 2026 so they compile and run on Mojo 1.2.0. The August 2025 drafts used APIs that have since moved, and each `README.md` ends with a short list of what changed. The programs use the Mojo already on your `PATH`. That compiler needs to be 1.2.0, and the axis-sum post also needs a build that includes MAX. They do not use the puzzle, manual, or Advent of Mojo environments.

| Post | Programs | What it covers |
|---|---|---|
| [simd-matrix](blog/simd-matrix/) | `simd_matrix.mojo` | CPU matrix multiplication in three steps: a scalar loop, `SIMD[DType.float32, 4]` vectorization, then a blocked loop. The post uses `ImmPointer` and `MutPointer`, indexes them with `unsafe_offset`, and reports benchmark numbers from a real run |
| [parallel-axis-sum](blog/parallel-axis-sum/) | `axis0_sum.mojo`, `axis1_sum.mojo` | GPU shared-memory reductions. `axis0_sum.mojo` sums down each column. `axis1_sum.mojo` sums across each row. Both were compiled and run on an Apple M3 Pro |

```bash
cd blog/simd-matrix
mojo simd_matrix.mojo
```

```bash
cd blog/parallel-axis-sum
mojo axis0_sum.mojo
mojo axis1_sum.mojo
```

## Resources

- [Mojo GPU Puzzles](https://puzzles.modular.com/)
- [Advent of Mojo](https://advent.mojolang.org/)
- [Mojo manual](https://mojolang.org/docs/manual/)
- [GPU programming fundamentals](https://docs.modular.com/mojo/manual/gpu/fundamentals)
