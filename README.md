# modlr

<p align="center">
  <img src="https://miro.medium.com/0*wS9xIg1uRHIbP4OP.jpeg" alt="Modular, the next-generation AI developer platform" width="690">
</p>

Speed is the hard part of modern AI software. Python is comfortable to write, and C++ or CUDA is what people reach for when a loop or a kernel has to keep up with the hardware. [Mojo](https://mojolang.org/), from [Modular](https://www.modular.com/), is built to cover both. Its syntax stays close to Python, and the compiler adds static types, SIMD, control over memory, and GPU kernels in the same source file.

Mojo is worth learning because a program can grow from a short example into a kernel that [MAX](https://docs.modular.com/max/) serves, written in one language throughout. This repository is a hands-on introduction, divided into four independent tracks.

| Track | Source | Mojo | Status |
|---|---|---|---|
| [problems/](#gpu-puzzles) | [Mojo GPU Puzzles](https://puzzles.modular.com/) | 1.2.0 | p01–p14 written, p15 not started |
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

Mojo is installed per folder, through [Pixi](https://pixi.sh/). There is no environment at the repository root. `problems/`, `advent-of-code/`, and `manual/` each download **Mojo 1.2.0** from the [stable MAX channel](https://conda.modular.com/max) the first time you run `pixi install` in that folder. Every track in this repository is meant to be run on that version.

### Install

1. Install Pixi, then open a new terminal so `pixi` is on your `PATH`.

   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh
   ```

2. Install one track and check the compiler. `manual/` is the smallest folder to start with.

   ```bash
   cd manual
   pixi install
   pixi run mojo --version
   ```

   The last command should print `Mojo 1.2.0`. `pixi install` is once per folder. After that, `pixi run mojo` uses the binary from that folder's lock file. Repeat it inside `advent-of-code/`, or inside a single `problems/pXX/` directory, when you move to that track. The tracks do not share an environment, so always `cd` into the folder you are about to run.

Each GPU puzzle also installs MAX (`max = "26.7.*"`). A plain `mojo` binary does not include `max.gpu`, so run those kernels with that puzzle's own `pixi run mojo`.

`blog/` has no Pixi project. From `manual/`, `pixi shell` puts this same Mojo 1.2.0 on your `PATH`, which is what the blog programs expect:

```bash
cd manual
pixi shell
mojo --version
```

### Requirements

- **Apple Silicon.** Every `pixi.toml` lists `osx-arm64`. Mojo on macOS wants macOS Sequoia (15) or later and the Xcode Command Line Tools. On Linux, add your platform to `platforms` and run `pixi install` again so Pixi rewrites the lock file.
- **Editor support.** The [Mojo extension for VS Code](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo) adds syntax highlighting and completion. The publisher is Modular.

## GPU Puzzles

Solutions to [Mojo GPU Puzzles](https://puzzles.modular.com/). The sequence starts with one thread writing one output element, then adds bounds checks, 2D indexing, blocks, shared memory, and full kernels: pooling, dot product, convolution, prefix sum, and matrix multiplication.

Each `problems/pXX/` directory is its own Pixi project (`pixi.toml` and `pixi.lock`). A kernel file is a GPU function plus a `main` that allocates device buffers with `DeviceContext`, launches the kernel, and checks the result with `assert_equal`.

### Version

Every puzzle runs on **Mojo 1.2.0**, the same compiler as `manual/`, `advent-of-code/`, and `blog/`. Each `pixi.toml` points at the [stable MAX channel](https://conda.modular.com/max) and pins both packages:

| Channel | `pixi.toml` | What that means |
|---|---|---|
| [stable MAX](https://conda.modular.com/max) | `mojo = "1.2.*"`, `max = "26.7.*"` | The lock resolves to Mojo 1.2.0 and MAX 26.7.0 |

`mojo` by itself does not ship `max.gpu`, so a 1.2.0 environment without `max` cannot compile these kernels. Install and run each puzzle from its own folder.

The 1.2.0 kernels import from `max.gpu` and `max.gpu.host`. They use `def`, `comptime`, and `Pointer[..., MutAnyOrigin]` indexed with `unsafe_offset`. Launches use `enqueue_function`, and scalar sizes passed into a kernel are `Int32`. p04 keeps a pointer kernel (`p04.mojo`) next to `p04-tile-tensor-intro.mojo` and `p04-tile-tensor.mojo`. p05 is the broadcast kernel, and it indexes the operands as `a[0, col] + b[row, 0]`.

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
| [p06](problems/p06/) | One block grid | `p06.mojo` |
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
