# Puzzle 06: Blocks

Add 10 to a vector that no longer fits in a single block.

Up to now every launch was one block, so `thread_idx` alone identified the element. Here the grid has **several blocks**, and `thread_idx.x` restarts at 0 in each one. A thread has to combine its block number with its position inside the block to find its **global** index.

## At a glance

| | |
|---|---|
| **Goal** | `output[i] = a[i] + 10` for a 9-element vector |
| **Data** | 1D vector, `SIZE = 9` floats |
| **Launch** | `grid_dim = (3, 1)` blocks, `block_dim = (4, 1)` threads per block |
| **Threads launched** | 3 × 4 = 12 (9 do work, 3 idle) |
| **New ideas** | multiple blocks, `block_idx`, `block_dim`, global index |
| **File** | `p06.mojo` |

## Why more than one block

A single block has a hardware limit on how many threads it can hold (commonly 1024), and all of a block's threads run on one GPU core cluster. Real data is far larger than that, and a large GPU has many core clusters to keep busy. So work is split into **many blocks**, which the GPU can schedule independently and in parallel.

The sizes here are tiny on purpose (blocks of 4 for 9 elements) so you can trace every thread by hand. The arithmetic is the same at any scale.

## The global index

```mojo
var i = block_dim.x * block_idx.x + thread_idx.x
```

| Name | Meaning | Values here |
|---|---|---|
| `block_dim.x` | Threads per block along x. The same for every block. | 4 |
| `block_idx.x` | Which block this thread is in. | 0, 1, 2 |
| `thread_idx.x` | Position inside that block. Restarts at 0 in every block. | 0, 1, 2, 3 |

`block_dim.x * block_idx.x` is the **number of elements owned by earlier blocks**. Block 2 skips the 8 elements that blocks 0 and 1 handle, and starts at 8. Adding `thread_idx.x` gives the offset within this block's slice.

It works like house numbers on a street of 4-house blocks: house 3 on block 1 is house `4 × 1 + 3 = 7` overall.

## Which thread does what

```
             block 0                block 1                block 2
       ┌────┬────┬────┬────┐  ┌────┬────┬────┬────┐  ┌────┬────┬────┬────┐
local  │ 0  │ 1  │ 2  │ 3  │  │ 0  │ 1  │ 2  │ 3  │  │ 0  │ 1  │ 2  │ 3  │
global │ 0  │ 1  │ 2  │ 3  │  │ 4  │ 5  │ 6  │ 7  │  │ 8  │ 9  │ 10 │ 11 │
       └────┴────┴────┴────┘  └────┴────┴────┴────┘  └────┴────┴────┴────┘
         ▼    ▼    ▼    ▼       ▼    ▼    ▼    ▼       ▼   ╳    ╳    ╳
 a     [ 0    1    2    3       4    5    6    7       8 ]   (i ≥ 9: skipped)
 out   [10   11   12   13      14   15   16   17      18 ]
```

| Block | `thread_idx.x` | Global `i` | `i < 9`? | Writes |
|---:|---|---|---|---|
| 0 | 0, 1, 2, 3 | 0, 1, 2, 3 | yes | `output[0..3] = 10..13` |
| 1 | 0, 1, 2, 3 | 4, 5, 6, 7 | yes | `output[4..7] = 14..17` |
| 2 | 0 | 8 | yes | `output[8] = 18` |
| 2 | 1, 2, 3 | 9, 10, 11 | no | nothing |

Block 2 is why the guard from [puzzle 03](../p03/README.md) is still needed. The grid is a rectangle of whole blocks, and the data length is not a multiple of the block size.

## Choosing the grid size

Given a data length `N` and a block size `B`, the number of blocks you need is the ceiling of `N / B`:

```
blocks = (N + B - 1) // B        # integer ceiling division
       = (9 + 4 - 1) // 4
       = 12 // 4 = 3
```

Plain `N // B = 2` would launch only 8 threads and leave `output[8]` unwritten. Always round **up**, then guard.

## The kernel

```mojo
comptime SIZE = 9
comptime BLOCKS_PER_GRID = (3, 1)
comptime THREADS_PER_BLOCK = (4, 1)
comptime dtype = DType.float32

def add_10_blocks(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var i = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 2 lines)
    if i < size:
        output[unsafe_offset=i] = a[unsafe_offset=i] + 10.0
```

The only change from puzzle 03 is the line that computes `i`. The guard and the body are the same. The kernel now also imports `block_idx` and `block_dim` from `max.gpu`.

`grid_dim` is now a tuple, `(3, 1)`, in the same `(x, y)` form as `block_dim`. The y parts are 1, so this is still a 1D problem. [Puzzle 07](../p07/README.md) uses both.

## What `main` does

`main` is the same as puzzle 03 apart from the sizes:

| Step | What happens |
|---|---|
| 1 | Allocate `out` and `a`, 9 floats each, and zero them. |
| 2 | Map `a` and write `0, 1, …, 8`. |
| 3 | Launch `add_10_blocks` with `Int32(SIZE)`, `grid_dim=(3, 1)`, and `block_dim=(4, 1)`. |
| 4 | Allocate and zero `expected`, synchronize, then write `expected[i] = i + 10`. |
| 5 | Map `out`, print both, and `assert_equal` all 9 elements. |

## Expected output

The first line names the device the kernel ran on: `metal` means the Apple GPU, `cuda` means an NVIDIA GPU (for example on Colab), and `cpu` would mean it ran on the CPU. The device name will differ on other machines.

```
Running on: Apple M3 Pro (metal, device 0)
out: HostBuffer([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
Puzzle 06 complete ✅
```

## Common mistakes

- **Using `thread_idx.x` alone.** Every block then writes elements 0 to 3. Elements 4 to 8 stay 0, and blocks overwrite each other's results.
- **`block_idx.x * thread_idx.x`.** A product instead of the formula. Every thread in block 0 gets `i = 0`.
- **Multiplying by the grid size.** `grid_dim.x * block_idx.x + thread_idx.x` uses the number of blocks (3) where it needs the block width (4). Indices collide and leave gaps.
- **Rounding the grid down.** `BLOCKS_PER_GRID = (2, 1)` leaves the last element unprocessed.
- **Dropping the guard because "the grid is exactly right".** Block sizes are usually fixed and data sizes are not. Keep the guard.

## Try it yourself

1. Set `BLOCKS_PER_GRID = (2, 1)`. Only 8 threads run, and the check fails at index 8.
2. Set `THREADS_PER_BLOCK = (3, 1)` and `BLOCKS_PER_GRID = (3, 1)`. Now 9 threads cover 9 elements exactly and none is idle.
3. Set `SIZE = 20`, keep 4 threads per block, and compute the block count with the ceiling formula. Check that `(20 + 4 - 1) // 4 = 5` passes and `4` does not.
4. Replace `i` with `thread_idx.x` and look at which elements come out wrong.

## Key takeaways

- With more than one block, `thread_idx` is **local**. The global index is `block_dim.x * block_idx.x + thread_idx.x`.
- Blocks are independent units of work that the GPU spreads across its cores. Large problems use many of them.
- The number of blocks is `ceil(N / block_size)`. Round up, then guard with `i < size`.

## Run

This folder is its own Pixi project. It pins Mojo 1.2.0 and MAX 26.7.0 on the stable MAX channel.

```bash
cd problems/p06
pixi install
pixi run mojo p06.mojo
```

`pixi install` is once.

### Run on a Colab GPU

The project also has tasks that run the puzzle on a Google Colab GPU, using the [Google Colab CLI](https://developers.googleblog.com/en/introducing-the-google-colab-cli/) that `pixi install` sets up. Every puzzle folder uses the same Colab machine, named `mojo`, so you start it once and then run any puzzle on it.

```bash
pixi run colab-up            # start a T4 GPU machine, install Mojo 1.2.0 + MAX 26.7.0 on it
pixi run colab-run           # upload p06.mojo and run it there
pixi run colab-status        # running machines and compute units left
pixi run colab-down          # stop the machine when you're done
```

- The first `colab` command asks you to sign in to Google. On the consent screen, tick every permission. If only your name and email are granted, the CLI stops with a "Scope has changed" error.
- `pixi run colab-up L4` picks another GPU. L4, A100, H100 and G4 need a paid Colab plan.
- Run `colab-up` once. A second run while the machine is up tries to create another machine with the same name. Check with `pixi run colab-status`.
- `colab-down` matters: a running machine keeps using your Colab quota.
- On Colab, the first output line names an NVIDIA GPU with `cuda` instead of `metal`, which confirms the puzzle ran there.

---

Previous: [Puzzle 05: Broadcast add](../p05/README.md) · Next: [Puzzle 07: 2D blocks](../p07/README.md)
