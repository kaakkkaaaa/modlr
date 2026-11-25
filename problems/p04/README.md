# Puzzle 04: 2D map

Add 10 to every element of a small matrix.

There are two new ideas. Threads are now arranged in two dimensions, so each one has a row and a column. And memory is still one flat array, so `(row, col)` has to be turned into a single offset with **row-major** indexing. The folder solves the puzzle twice: once with raw pointers and manual offsets, and once with `TileTensor`, which does the offset math for you.

## At a glance

| | |
|---|---|
| **Goal** | `output[row, col] = a[row, col] + 10` |
| **Data** | 2×2 matrix (`SIZE = 2`), stored as 4 contiguous floats |
| **Launch** | `grid_dim = 1` block, `block_dim = (3, 3)` threads |
| **Threads launched** | 9 (4 do work, 5 idle) |
| **New ideas** | `thread_idx.y`, 2D guard, row-major offsets, `TileTensor`, `row_major` layout |
| **Files** | `p04.mojo`, `p04-tile-tensor.mojo` |

## 2D threads

A block can be shaped in two (or three) dimensions. `block_dim = (3, 3)` makes a 3×3 grid of threads, and each thread has both `thread_idx.x` and `thread_idx.y`:

```
             thread_idx.x →
               0       1       2
            ┌───────┬───────┬───────┐
 thread_    │ (0,0) │ (1,0) │ (2,0) │  0
 idx.y      ├───────┼───────┼───────┤
   ↓        │ (0,1) │ (1,1) │ (2,1) │  1
            ├───────┼───────┼───────┤
            │ (0,2) │ (1,2) │ (2,2) │  2
            └───────┴───────┴───────┘
                labels are (x, y)
```

The kernel maps **y to the row** and **x to the column**:

```mojo
var row = thread_idx.y
var col = thread_idx.x
```

That choice is deliberate. Threads next to each other along x get columns next to each other, and in row-major storage neighbouring columns are neighbouring addresses. GPUs read memory fastest when neighbouring threads read neighbouring addresses (this is called *coalescing*). Using x for the column keeps that true.

## Row-major storage

Device memory is one-dimensional. A matrix is stored by laying its rows end to end:

```
 matrix view            memory (flat)
 ┌─────┬─────┐
 │  0  │  1  │  row 0   [ 0 , 1 , 2 , 3 ]
 ├─────┼─────┤            └─row 0─┘└─row 1─┘
 │  2  │  3  │  row 1
 └─────┴─────┘
```

To reach `(row, col)`, skip `row` complete rows of `SIZE` elements each, then move `col` more:

```
offset = row * SIZE + col
```

| `(row, col)` | Offset `row * 2 + col` | Input | Output |
|---|---:|---:|---:|
| `(0, 0)` | 0 | 0 | 10 |
| `(0, 1)` | 1 | 1 | 11 |
| `(1, 0)` | 2 | 2 | 12 |
| `(1, 1)` | 3 | 3 | 13 |

## The 2D guard

The block is 3×3 = 9 threads, and the matrix is only 2×2 = 4 elements. Any thread with `row >= 2` **or** `col >= 2` is outside the matrix. The puzzle 03 guard now has to hold on both axes:

```mojo
if row < size and col < size:
```

```
               col 0   col 1   col 2
            ┌───────┬───────┬───────┐
   row 0    │ work  │ work  │ idle  │
            ├───────┼───────┼───────┤
   row 1    │ work  │ work  │ idle  │
            ├───────┼───────┼───────┤
   row 2    │ idle  │ idle  │ idle  │
            └───────┴───────┴───────┘
```

Checking only `row` is not enough. Thread `(row=0, col=2)` would compute offset `0 * 2 + 2 = 2` and overwrite the element that belongs to `(1, 0)`. In 2D, an index that is out of range on one axis can land *inside* the buffer at the wrong element. That is worse than a crash, because nothing tells you it happened.

## Version 1: raw pointers (`p04.mojo`)

```mojo
def add_10_2d(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var row = thread_idx.y
    var col = thread_idx.x
    # FILL ME IN (roughly 2 lines)
    if row < size and col < size:
        output[unsafe_offset=(row * size + col)] = a[unsafe_offset=(row * size + col)] + 10.0
```

You compute the flat offset yourself. The host side fills the input with a nested `(i, j)` loop that uses the same `i * SIZE + j` formula, and checks the output with it too, so the host and the kernel agree on the layout.

## Version 2: `TileTensor`

### What a `TileTensor` is

A `TileTensor` is a **view**: a pointer to existing memory plus a **layout** that maps logical coordinates to offsets. It does not own or copy the data.

```mojo
from layout import TileTensor
from layout.tile_layout import row_major

comptime layout = row_major[SIZE, SIZE]()   # shape (2, 2), row-major strides
comptime LayoutType = type_of(layout)

var a_tensor = TileTensor(a, layout)        # wrap an existing DeviceBuffer
```

`row_major[SIZE, SIZE]()` is the layout. It knows the shape and computes `row * SIZE + col` itself, so `a_tensor[row, col]` reaches the right element without you writing the formula. The layout is a `comptime` value, which means its shape is part of the tensor's **type**. The tensor wraps device memory, so the host checks the underlying buffer after `synchronize()` rather than printing the tensor from `main`.

### `p04-tile-tensor.mojo`

```mojo
def add_10_2d(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var row = thread_idx.y
    var col = thread_idx.x
    # FILL ME IN (roughly 2 lines)
    if row < size and col < size:
        output[row, col] = a[row, col] + 10.0
```

The thread mapping and the guard are the same as version 1. Only the indexing changed: `output[row, col]` instead of `output[unsafe_offset=(row * size + col)]`.

In `main`, each `DeviceBuffer` is wrapped once with `TileTensor(buffer, layout)`, and the tensors are passed to `enqueue_function`. The host prints the shape with `out_tensor.dim[0]()` and `out_tensor.dim[1]()`, then checks the **underlying flat buffer** (`out_buf`) element by element. The tensor and the buffer refer to the same memory.

`TileTensor` does not do bounds checking for you. The guard is still required.

### Pointer vs. `TileTensor`

| | Raw `Pointer` | `TileTensor` |
|---|---|---|
| Indexing | `p[unsafe_offset=row * size + col]` | `t[row, col]` |
| Shape known to the kernel? | No | Yes, part of the type |
| Who computes the offset | You | The layout |
| Changing the layout | Rewrite every index expression | Change the `layout` value |
| Bounds checking | None | None. You still need the guard |

## Expected output

The first line names the device the kernel ran on: `metal` means the Apple GPU, `cuda` means an NVIDIA GPU (for example on Colab), and `cpu` would mean it ran on the CPU. The device name will differ on other machines.

`p04.mojo`:

```
Running on: Apple M3 Pro (metal, device 0)
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
Puzzle 04 complete ✅
```

`p04-tile-tensor.mojo`:

```
Running on: Apple M3 Pro (metal, device 0)
out shape: 2 x 2
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
Puzzle 04 complete ✅
```

## Common mistakes

- **Swapping x and y.** `row = thread_idx.x, col = thread_idx.y` gives the right answer on a square matrix with this test, but neighbouring threads then read addresses `SIZE` apart, which loses coalescing. On a non-square matrix it gives wrong indices.
- **`col * SIZE + row`.** That is column-major order. It disagrees with how the host filled the buffer.
- **Guarding one axis only.** See [The 2D guard](#the-2d-guard). Out-of-range columns land on valid memory in the next row.
- **Using `or` in the guard.** `row < size or col < size` lets `(0, 2)` through. Both conditions must hold.

## Try it yourself

1. In `p04.mojo`, change the guard to `if row < size:` and run it. Look at which element is wrong, and work out which thread overwrote it.
2. Change `SIZE` to 3. The 3×3 block now fits exactly and no thread is idle. Then try `SIZE = 4`: the block is too small, and the last row and column stay 0.
3. In `p04-tile-tensor.mojo`, change the layout to `row_major[SIZE, SIZE]()` with `SIZE = 3` and keep the `(3, 3)` block. Every thread now has an element. Then set `SIZE = 4` and watch the last row and column stay 0.

## Key takeaways

- Blocks can be 2D. `thread_idx.y` gives the row and `thread_idx.x` gives the column.
- Memory is flat. Row-major offset = `row * width + col`.
- Guard **every** axis. In 2D, a bad index can hit valid memory at the wrong element.
- `TileTensor` is a typed view (pointer plus layout) that does the offset math for you, but not the bounds check.

## Run

This folder is its own Pixi project. It pins Mojo 1.2.0 and MAX 26.7.0 on the stable MAX channel.

```bash
cd problems/p04
pixi install
pixi run mojo p04.mojo
pixi run mojo p04-tile-tensor.mojo
```

`pixi install` is once.

### Run on a Colab GPU

The project also has tasks that run the puzzle on a Google Colab GPU, using the [Google Colab CLI](https://developers.googleblog.com/en/introducing-the-google-colab-cli/) that `pixi install` sets up. Every puzzle folder uses the same Colab machine, named `mojo`, so you start it once and then run any puzzle on it.

```bash
pixi run colab-up            # start a T4 GPU machine, install Mojo 1.2.0 + MAX 26.7.0 on it
pixi run colab-run           # upload p04.mojo and run it there
pixi run colab-run p04-tile-tensor.mojo
pixi run colab-status        # running machines and compute units left
pixi run colab-down          # stop the machine when you're done
```

- The first `colab` command asks you to sign in to Google. On the consent screen, tick every permission. If only your name and email are granted, the CLI stops with a "Scope has changed" error.
- `pixi run colab-up L4` picks another GPU. L4, A100, H100 and G4 need a paid Colab plan.
- Run `colab-up` once. A second run while the machine is up tries to create another machine with the same name. Check with `pixi run colab-status`.
- `colab-down` matters: a running machine keeps using your Colab quota.
- On Colab, the first output line names an NVIDIA GPU with `cuda` instead of `metal`, which confirms the puzzle ran there.

---

Previous: [Puzzle 03: Guard](../p03/README.md) · Next: [Puzzle 05: Broadcast add](../p05/README.md)
