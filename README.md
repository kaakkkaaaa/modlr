# modlr

My notes, exercises, and solutions while learning the [Mojo](https://mojolang.org/) programming language, split into three independent tracks.

| Track | Source | Mojo | Status |
|---|---|---|---|
| [problems/](#problems--gpu-puzzles) | [Mojo GPU Puzzles](https://puzzles.modular.com/) | 0.25.7 (nightly) | p01–p14 written, p15 not started |
| [advent-of-code/](#advent-of-code--advent-of-mojo) | [Advent of Mojo](https://advent.mojolang.org/) | 1.2.x | Chapters 1–8 |
| [manual/](#manual--mojo-manual-notes) | [Mojo manual](https://mojolang.org/docs/manual/) | 1.2.0 | Part 1 mostly done (steps 1.1–1.11) |

```
.
├── problems/         # one Pixi env per puzzle
├── advent-of-code/   # one shared Pixi env
└── manual/           # one shared Pixi env
```

## Before you start

- **Install [Pixi](https://pixi.sh/).** Every track manages Mojo through Pixi. There is no environment at the repository root, and the tracks do not share a Mojo version, so always `cd` into a track (or puzzle) folder first.
- **Apple Silicon only.** Every `pixi.toml` targets `osx-arm64`. On another platform, add your platform to `platforms` and run `pixi install` to regenerate the lock file.

## problems/: GPU puzzles

Solutions to [Mojo GPU Puzzles](https://puzzles.modular.com/). The sequence starts with one thread writing one output element, then adds bounds checks, 2D indexing, blocks, shared memory, and full kernels: pooling, dot product, convolution, prefix sum, and matrix multiplication.

Each `problems/pXX/` directory is its own Pixi project (`pixi.toml` and `pixi.lock`) on the [MAX nightly channel](https://conda.modular.com/max-nightly). A kernel file is a GPU function plus a `main` that allocates device buffers with `DeviceContext`, launches the kernel, and checks the result with `assert_equal`. From p04 onward, most puzzles keep a raw-pointer version and a `LayoutTensor` version side by side.

### Version

> [!NOTE]
> This track still targets **Mojo 0.25.7**. It will be updated to **Mojo 1.2.0**, the same version already used by `manual/` and `advent-of-code/`.

The kernels were written against nightly builds from before the 1.2 release. Two pins are in the tree today:

| Folders | `pixi.toml` constraint | What that means |
|---|---|---|
| p01–p07 | `mojo = "0.25.7.*"` | Any 0.25.7 release on the nightly channel |
| p08–p15 | `mojo = ">=0.25.7.0.dev2025100505,<0.26"` | The older nightly these folders were first locked to |

The 1.2.0 update will do three things:

1. Point every puzzle at the stable MAX channel and pin `mojo = "1.2.*"`, matching `manual/pixi.toml`.
2. Bring the remaining 0.25.7 GPU API in line with 1.2. Later puzzles still use `alias` for compile-time values, `UnsafePointer` without an origin, and `enqueue_function`. Earlier puzzles have already moved toward `comptime`, `MutAnyOrigin`, and `enqueue_function_checked`.
3. Re-lock each `pixi.lock` on `osx-arm64` after the code compiles and the existing `assert_equal` checks pass.

Until that lands, install and run a puzzle with the Pixi environment inside its own folder. The 1.2 environments under `manual/` and `advent-of-code/` are a different compiler, so leave them for those tracks.

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
| [p04](problems/p04/) | 2D map, then the same kernel with `LayoutTensor` | `p04.mojo`, `p04-layout-tensor*.mojo` |
| [p05](problems/p05/) | Broadcast add | `p05.mojo`, `p05-layout-tensor.mojo` |
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

## advent-of-code/: Advent of Mojo

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

## manual/: Mojo manual notes

Study notes for the Mojo **1.2.0** manual, one file per step. The environment (`manual/pixi.toml`) pins `mojo = "1.2.*"` and adds Python 3.12 and NumPy for Part 8, where Mojo calls Python. [manual/README.md](manual/README.md) is the progress table for all eight parts. Each part folder has its own README with the step checklist, goals, and checkpoint questions.

Part 1 is the only part with programs so far: `s01_hello.mojo` through `s11_match.mojo`. Those files are small programs you can run, covering `main`, `var` and `comptime`, numbers, SIMD, strings, tuples, `Array` and `List`, `Dict`, `Set`, `Optional`, operators, loops, and `match`. Steps 1.12–1.14 (functions, raising, and a closing exercise) are still unwritten. Parts 2–8 are checklists waiting for the same treatment: structs and traits, ownership, lifecycle, metaprogramming, pointers, advanced functions, and Python/C interop.

```bash
cd manual
pixi install
pixi run version                                              # prints the Mojo version
pixi run mojo run p01_basics/s01_hello.mojo                   # run a file
pixi run mojo build p01_basics/s01_hello.mojo -o build/hello  # build a binary
```

| Part | Topic | Status |
|---|---|---|
| [p01_basics](manual/p01_basics/) | Language basics | `s01`–`s11` written. Steps 1.12–1.14 to do |
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

</details>

## Resources

- [Mojo GPU Puzzles](https://puzzles.modular.com/)
- [Advent of Mojo](https://advent.mojolang.org/)
- [Mojo manual](https://mojolang.org/docs/manual/)
- [GPU programming fundamentals](https://docs.modular.com/mojo/manual/gpu/fundamentals)
