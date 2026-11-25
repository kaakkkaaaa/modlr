# Puzzle 07: 2D blocks

Add 10 to every element of a 5×5 matrix, using more than one block along **both** axes.

This puzzle puts the previous ones together: the `(row, col)` indexing and 2D guard from [puzzle 04](../p04/README.md), the global-index formula from [puzzle 06](../p06/README.md), and the generic `TileTensor` signature from [puzzle 05](../p05/README.md). It solves the problem twice, once with `TileTensor` and once with raw pointers, and checks both.

## At a glance

| | |
|---|---|
| **Goal** | `output[row, col] = a[row, col] + 10` |
| **Data** | 5×5 matrix (`SIZE = 5`), 25 floats, row-major |
| **Launch** | `grid_dim = (2, 2)` blocks, `block_dim = (3, 3)` threads per block |
| **Threads launched** | 4 blocks × 9 = 36 (25 do work, 11 idle) |
| **New ideas** | 2D grid of blocks, global row and column, two kernels sharing one input |
| **File** | `p07.mojo` |

## The global coordinates

Puzzle 06's formula is applied to each axis separately:

```mojo
var row = block_dim.y * block_idx.y + thread_idx.y
var col = block_dim.x * block_idx.x + thread_idx.x
```

| Axis | Block size | Block index | Thread index | Global |
|---|---|---|---|---|
| y (rows) | `block_dim.y = 3` | `block_idx.y ∈ {0, 1}` | `thread_idx.y ∈ {0, 1, 2}` | `row ∈ 0..5` |
| x (columns) | `block_dim.x = 3` | `block_idx.x ∈ {0, 1}` | `thread_idx.x ∈ {0, 1, 2}` | `col ∈ 0..5` |

As in puzzle 04, **y is the row and x is the column**, so neighbouring threads along x touch neighbouring addresses in row-major memory.

## Sizing the grid

Each axis needs `ceil(5 / 3) = (5 + 3 - 1) // 3 = 2` blocks. The 2×2 grid of 3×3 blocks covers a 6×6 thread rectangle, one row and one column more than the 5×5 matrix:

```
                  block_idx.x = 0          block_idx.x = 1
                 col 0   col 1   col 2 │ col 3   col 4   col 5
               ┌───────┬───────┬───────┼───────┬───────┬───────┐
        row 0  │   0   │   1   │   2   │   3   │   4   │   ·   │
block_  row 1  │   5   │   6   │   7   │   8   │   9   │   ·   │
idx.y=0 row 2  │  10   │  11   │  12   │  13   │  14   │   ·   │
               ├───────┼───────┼───────┼───────┼───────┼───────┤
        row 3  │  15   │  16   │  17   │  18   │  19   │   ·   │
block_  row 4  │  20   │  21   │  22   │  23   │  24   │   ·   │
idx.y=1 row 5  │   ·   │   ·   │   ·   │   ·   │   ·   │   ·   │
               └───────┴───────┴───────┴───────┴───────┴───────┘
   numbers = flat index (row * 5 + col) of the element that thread handles
   ·       = thread outside the matrix, skipped by the guard
```

| `block_idx (x, y)` | Columns | Rows | Active threads |
|---|---|---|---:|
| `(0, 0)` | 0, 1, 2 | 0, 1, 2 | 9 |
| `(1, 0)` | 3, 4 (5 skipped) | 0, 1, 2 | 6 |
| `(0, 1)` | 0, 1, 2 | 3, 4 (5 skipped) | 6 |
| `(1, 1)` | 3, 4 (5 skipped) | 3, 4 (5 skipped) | 4 |
| **Total** | | | **25** |

The guard `if row < size and col < size` drops the 11 threads in the extra row and column. As puzzle 04 showed, checking only one axis is a real bug: thread `(row=0, col=5)` would compute flat offset `0 * 5 + 5 = 5` and overwrite element `(1, 0)`.

## Worked example

Take the thread with `block_idx = (1, 1)` and `thread_idx = (1, 0)` (x, y):

```
row = block_dim.y * block_idx.y + thread_idx.y = 3 * 1 + 0 = 3
col = block_dim.x * block_idx.x + thread_idx.x = 3 * 1 + 1 = 4
guard: 3 < 5 and 4 < 5  → passes
flat offset = 3 * 5 + 4 = 19
output[3, 4] = a[3, 4] + 10 = 19 + 10 = 29
```

## Two kernels, one input

`p07.mojo` runs the same add twice, reading the same input buffer `a` and writing two separate output buffers.

### Kernel 1: `add_10_blocks_2d` (TileTensor)

```mojo
def add_10_blocks_2d[
    Engine: TensorEngine
](
    output: TileTensor[mut=True,  dtype, OutLayout, MutAnyOrigin,   Engine=Engine],
    a:      TileTensor[mut=False, dtype, ALayout,   ImmutAnyOrigin, Engine=Engine],
    size_dev: Int32,
) where (Engine.element_size == 1):
    var size = Int(size_dev)
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 2 lines)
    if row < size and col < size:
        output[row, col] = a[row, col] + 10.0
```

This uses the same signature as puzzle 05: generic over `Engine`, `element_size == 1`, writable `output`, read-only `a`. Both layouts are `row_major[SIZE, SIZE]()`. `main` picks the concrete engine and gives the specialized kernel a name before launching it:

```mojo
comptime tile_tensor_kernel = add_10_blocks_2d[type_of(output_tensor).Engine]
ctx.enqueue_function[tile_tensor_kernel](output_tensor, a_tensor, Int32(SIZE), ...)
```

`type_of(output_tensor).Engine` and puzzle 05's `out_tensor.Engine` name the same thing. The `comptime` alias just makes the launch line shorter.

### Kernel 2: `add_10_blocks_2d_raw_mem` (raw pointers)

```mojo
def add_10_blocks_2d_raw_mem(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 2 lines)
    if row < size and col < size:
        output[unsafe_offset=(row * size + col)] = a[unsafe_offset=(row * size + col)] + 10.0
```

The coordinates and the guard are identical. The only difference is that you compute the row-major offset `row * size + col` yourself, as in `p04.mojo`.

Writing both side by side shows what `TileTensor` saves you (the offset formula) and what it does not (the global index and the guard).

## What `main` does

| Step | What happens |
|---|---|
| 1 | Allocate the shared input `a` (25 floats) and write `a[i] = i`, so `a[row, col] = row * 5 + col`. Every element is written, so no `enqueue_fill` is needed first. |
| 2 | Allocate and zero `output_tile`. Wrap it and `a` as `TileTensor`s. |
| 3 | Launch the TileTensor kernel. |
| 4 | Allocate and zero `output_raw`. Launch the raw-pointer kernel with the plain buffers. |
| 5 | `ctx.synchronize()` once, which waits for both launches. |
| 6 | `check(output_tile, "TileTensor")` and `check(output_raw, "Raw Memory")`. |

Both launches go on the same `DeviceContext`, so they run in the order they were queued. They both only read `a` and write different outputs, so they do not interfere.

`check` is a small helper so the verification is not written twice. It maps a buffer, prints it, and checks every `(i, j)` against `Scalar[dtype](i * SIZE + j + 10)`. It computes the expected value directly instead of building a separate `expected` buffer. Both checks must pass before the final line prints.

## Expected output

The first line names the device the kernel ran on: `metal` means the Apple GPU, `cuda` means an NVIDIA GPU (for example on Colab), and `cpu` would mean it ran on the CPU. The device name will differ on other machines.

```
Running on: Apple M3 Pro (metal, device 0)
--------------------------------
TileTensor : HostBuffer([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0])
✅ TileTensor passed
--------------------------------
Raw Memory : HostBuffer([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0])
✅ Raw Memory passed
--------------------------------
Puzzle 07 complete ✅
```

## Common mistakes

- **Mixing axes in the formula.** `row = block_dim.x * block_idx.y + thread_idx.y` happens to work here because the block is square (3×3). It breaks with a non-square block such as `(4, 2)`. Pair `.y` with `.y` and `.x` with `.x`.
- **Forgetting the block term on one axis.** `row = thread_idx.y` makes blocks with `block_idx.y = 1` redo rows 0 to 2, and rows 3 and 4 are never written.
- **Rounding the grid down.** `grid_dim = (1, 1)` covers only the top-left 3×3. Each axis needs `ceil(5 / 3) = 2` blocks.
- **Guarding one axis.** The extra column wraps into the next row in flat memory and overwrites a valid element.

## Try it yourself

1. Change `THREADS_PER_BLOCK` to `(5, 5)` and `BLOCKS_PER_GRID` to `(1, 1)`. One block now covers the matrix exactly, which is puzzle 04 again.
2. Use a non-square block: `THREADS_PER_BLOCK = (4, 2)`. Work out the grid you need (`ceil(5/4) = 2` columns of blocks, `ceil(5/2) = 3` rows of blocks, so `BLOCKS_PER_GRID = (2, 3)`) and check that both kernels still pass.
3. Set `BLOCKS_PER_GRID = (1, 2)` and predict which columns of the printed output stay at 0.
4. Remove `col < size` from the raw-pointer guard only. Compare its output with the TileTensor result and find the overwritten elements.

## Key takeaways

- In 2D, each axis gets its own global index: `row` from the y components and `col` from the x components.
- Size the grid per axis with `ceil(N / block_size)`. The launch covers a rectangle at least as large as the data.
- The 2D guard `row < size and col < size` is required whenever either axis is padded.
- `TileTensor` replaces the offset arithmetic, not the indexing logic. The global coordinates and the guard are the same in both kernels.

## Run

This folder is its own Pixi project. It pins Mojo 1.2.0 and MAX 26.7.0 on the stable MAX channel.

```bash
cd problems/p07
pixi install
pixi run mojo p07.mojo
```

`pixi install` is once.

### Run on a Colab GPU

The project also has tasks that run the puzzle on a Google Colab GPU, using the [Google Colab CLI](https://developers.googleblog.com/en/introducing-the-google-colab-cli/) that `pixi install` sets up. Every puzzle folder uses the same Colab machine, named `mojo`, so you start it once and then run any puzzle on it.

```bash
pixi run colab-up            # start a T4 GPU machine, install Mojo 1.2.0 + MAX 26.7.0 on it
pixi run colab-run           # upload p07.mojo and run it there
pixi run colab-status        # running machines and compute units left
pixi run colab-down          # stop the machine when you're done
```

- The first `colab` command asks you to sign in to Google. On the consent screen, tick every permission. If only your name and email are granted, the CLI stops with a "Scope has changed" error.
- `pixi run colab-up L4` picks another GPU. L4, A100, H100 and G4 need a paid Colab plan.
- Run `colab-up` once. A second run while the machine is up tries to create another machine with the same name. Check with `pixi run colab-status`.
- `colab-down` matters: a running machine keeps using your Colab quota.
- On Colab, the first output line names an NVIDIA GPU with `cuda` instead of `metal`, which confirms the puzzle ran there.

---

Previous: [Puzzle 06: Blocks](../p06/README.md) · Next: [Puzzle 08: Shared memory](../p08/README.md)
