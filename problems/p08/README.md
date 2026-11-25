# Puzzle 08: Shared memory

Add 10 to every element of an 8-element vector, but route each value through **shared memory** on the way.

The answer is the same as puzzle 06. What is new is where the data goes in between: each block copies its part of the input into a small scratchpad that all of its threads can see, waits for every thread to finish, and only then computes. This load → `barrier()` → compute pattern is the basis of the later puzzles. Like [puzzle 07](../p07/README.md), it solves the problem twice, once with `TileTensor` and once with raw pointers, and checks both.

## At a glance

| | |
|---|---|
| **Goal** | `output[i] = a[i] + 10`, reading `a[i]` through shared memory |
| **Data** | 8 floats (`SIZE = 8`) |
| **Launch** | `grid_dim = (2, 1)` blocks, `block_dim = (TPB, 1) = (4, 1)` threads per block |
| **Threads launched** | 2 blocks × 4 = 8 (all do work) |
| **Shared memory** | `TPB = 4` floats per block (16 bytes) |
| **New ideas** | shared memory, `stack_allocation` with `AddressSpace.SHARED`, `barrier()`, local vs. global index |
| **File** | `p08.mojo` |

## Global memory vs. shared memory

So far every kernel read and wrote **global memory**: the buffers created with `ctx.enqueue_create_buffer`. Global memory is large and every thread in the launch can reach it, but it is the slowest memory on the GPU.

**Shared memory** is a small, fast scratchpad that lives on the GPU core running a block:

| | Global memory | Shared memory |
|---|---|---|
| Who can see it | every thread, every block | only the threads of **one** block |
| Lifetime | until the buffer is freed | one block's run of the kernel |
| Created by | the host (`enqueue_create_buffer`) | the kernel (`stack_allocation`) |
| Size | gigabytes | tens of kilobytes per block |
| Speed | slow | much faster |

Each block gets its **own** copy. Block 0's `shared[0]` and block 1's `shared[0]` are different memory, so both blocks can use slots 0 to 3 without clashing.

## Local index and global index

Two indexes are in play:

```mojo
var global_i = block_dim.x * block_idx.x + thread_idx.x   # position in the whole array (puzzle 06)
var local_i = thread_idx.x                                 # position inside this block
```

Global memory spans the whole array, so it is indexed with `global_i`. Shared memory holds only one block's slice, so it is indexed with `local_i`:

| `block_idx.x` | `thread_idx.x` | `local_i` | `global_i` | Reads `a[...]` | Uses `shared[...]` | Writes `output[...]` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| 0 | 2 | 2 | 2 | 2 | 2 | 2 |
| 0 | 3 | 3 | 3 | 3 | 3 | 3 |
| 1 | 0 | 0 | 4 | 4 | 0 (block 1's copy) | 4 |
| 1 | 1 | 1 | 5 | 5 | 1 (block 1's copy) | 5 |
| 1 | 2 | 2 | 6 | 6 | 2 (block 1's copy) | 6 |
| 1 | 3 | 3 | 7 | 7 | 3 (block 1's copy) | 7 |

## The three steps

Every thread runs the same three steps:

```
1. load     shared[local_i] = a[global_i]          global → shared
2. barrier  wait until every thread in the block has finished step 1
3. compute  output[global_i] = shared[local_i] + 10    shared → global
```

`barrier()` is a meeting point for the threads of **one block**. No thread in the block continues past it until all of them have reached it. After the barrier, every slot of `shared` is guaranteed to hold its value.

Strictly, this puzzle would work without the barrier, because each thread only reads back the slot it wrote itself. The barrier becomes essential as soon as a thread reads a slot **another** thread wrote, which is what the next puzzles do (pooling, convolution, reductions). It is here to build the habit.

`barrier()` only synchronizes threads within a block. Nothing synchronizes blocks with each other inside a kernel, which is one more reason each block keeps its own shared memory.

## Two kernels, one input

Both kernels need a function named `stack_allocation`, but from different modules: the raw-memory one from `std.memory`, the TileTensor one from `layout.tile_tensor`. Importing both under the same name would clash, so `p08.mojo` renames them with `as`:

```mojo
from std.memory import Pointer, stack_allocation as mem_stack_allocation
from layout.tile_tensor import stack_allocation as tensor_stack_allocation
```

### Kernel 1: `add_10_shared` (TileTensor)

```mojo
def add_10_shared[
    Engine: TensorEngine,
](
    output: TileTensor[mut=True,  dtype, LayoutType, MutAnyOrigin,   Engine=Engine],
    a:      TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin, Engine=Engine],
    size_dev: Int32,
) where (Engine.element_size == 1):
    var size = Int(size_dev)
    var shared = tensor_stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[TPB](),
    )
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    if global_i < size:
        shared[local_i] = rebind[Scalar[dtype]](a[global_i])
    barrier()
    if global_i < size:
        output[global_i] = shared[local_i] + 10
```

`tensor_stack_allocation` takes a **layout** (`row_major[TPB]()`) and returns a `TileTensor` that lives in shared memory, so it is indexed with `[local_i]` like any other tensor. Reading `a[global_i]` from a `TileTensor` gives a small vector type, so `rebind[Scalar[dtype]]` turns it into a plain scalar before storing it.

### Kernel 2: `add_10_shared_raw_mem` (raw pointers)

```mojo
def add_10_shared_raw_mem(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var shared = mem_stack_allocation[
        TPB, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    if global_i < size:
        shared[unsafe_offset=local_i] = a[unsafe_offset=global_i]
    barrier()
    # FILL ME IN (roughly 2 lines)
    if global_i < size:
        output[unsafe_offset=global_i] = shared[unsafe_offset=local_i] + 10.0
```

`mem_stack_allocation` takes an element **count** and an element **type** and returns a pointer into shared memory. A pointer read is already a `Scalar[dtype]`, so no `rebind` is needed.

| | TileTensor | Raw memory |
|---|---|---|
| Shared allocation | `tensor_stack_allocation[dtype=..., address_space=.SHARED](row_major[TPB]())` | `mem_stack_allocation[TPB, Scalar[dtype], address_space=AddressSpace.SHARED]()` |
| Indexing | `shared[local_i]`, `output[global_i]` | `shared[unsafe_offset=local_i]`, `output[unsafe_offset=global_i]` |
| Reading `a` | `rebind[Scalar[dtype]](a[global_i])` | `a[unsafe_offset=global_i]` |

The logic is identical: load at `local_i`, `barrier()`, write at `global_i`, with the same guard on both sides of the barrier.

## What `main` does

| Step | What happens |
|---|---|
| 1 | Print the device the program runs on. |
| 2 | Allocate the shared input `a` (8 floats) and write `a[i] = i`. Every element is written, so no `enqueue_fill` is needed first. |
| 3 | Allocate and zero `out_tile`. Wrap it and `a` as `TileTensor`s and launch the TileTensor kernel, specialized with `add_10_shared[out_tensor.Engine]`. |
| 4 | Allocate and zero `out_raw`. Launch the raw-pointer kernel with the plain buffers. |
| 5 | `ctx.synchronize()` once, which waits for both launches. |
| 6 | `check(out_tile, "TileTensor")` and `check(out_raw, "Raw Memory")`. |

`check` maps a buffer, prints it, and checks every element against `Scalar[dtype](i + 10)`. Using `a[i] = i` instead of a constant input means each position has a different correct answer, so a kernel that read the wrong shared slot or wrote the wrong position would fail.

## Expected output

The first line names the device the kernel ran on: `metal` means the Apple GPU, `cuda` means an NVIDIA GPU (for example on Colab), and `cpu` would mean it ran on the CPU. The device name will differ on other machines.

```
Running on: Apple M3 Pro (metal, device 0)
--------------------------------
TileTensor : HostBuffer([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
TileTensor passed ✅
--------------------------------
Raw Memory : HostBuffer([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
Raw Memory passed ✅
--------------------------------
Puzzle 8 complete ✅
```

## Common mistakes

- **Indexing shared memory with `global_i`.** Shared memory holds only `TPB = 4` slots. In block 1, `global_i` runs from 4 to 7, so `shared[global_i]` goes past the end of the allocation.
- **Indexing the output with `local_i`.** Both blocks would then write `output[0..3]`, and `output[4..7]` stays 0.
- **Putting `barrier()` inside the `if`.** Every thread of the block must reach the same `barrier()`. If some threads skip it, the block can hang or behave unpredictably. Keep the barrier outside any condition, as here.
- **Assuming the barrier syncs the whole launch.** It only waits for the threads of the current block. Block 1 cannot see block 0's shared memory at all.
- **Importing both `stack_allocation`s under one name.** The second import hides the first. Rename at least one with `as`.

## Try it yourself

1. Make the barrier matter: in the raw-memory kernel, compute `output[global_i] = shared[(local_i + 1) % TPB] + 10` so each thread reads its neighbour's slot, and update the expected values in `check`. Then comment out `barrier()` and see whether the result still holds.
2. Change `TPB` to 8 and `BLOCKS_PER_GRID` to `(1, 1)`. One block now covers the whole array, with an 8-slot shared buffer.
3. Change `SIZE` to 7 and keep the grid. The last thread of block 1 is now outside the data, and the guards on both sides of the barrier skip it.
4. Remove the guard from the load step only and keep `SIZE = 7`. Work out which global address the extra thread would read.

## Key takeaways

- Shared memory is a fast scratchpad **per block**, allocated inside the kernel with `stack_allocation` and `AddressSpace.SHARED`.
- Index shared memory with the **local** index (`thread_idx.x`) and global memory with the **global** index.
- `barrier()` makes every thread of a block wait until all have arrived. Use it between writing shared memory and reading slots written by other threads, and never put it inside a branch that only some threads take.
- The load → `barrier()` → compute pattern is the building block for the pooling, convolution and reduction puzzles that follow.

## Run

This folder is its own Pixi project. It pins Mojo 1.2.0 and MAX 26.7.0 on the stable MAX channel.

```bash
cd problems/p08
pixi install
pixi run mojo p08.mojo
```

`pixi install` is once.

### Run on a Colab GPU

The project also has tasks that run the puzzle on a Google Colab GPU, using the [Google Colab CLI](https://developers.googleblog.com/en/introducing-the-google-colab-cli/) that `pixi install` sets up. Every puzzle folder uses the same Colab machine, named `mojo`, so you start it once and then run any puzzle on it.

```bash
pixi run colab-up            # start a T4 GPU machine, install Mojo 1.2.0 + MAX 26.7.0 on it
pixi run colab-run           # upload p08.mojo and run it there
pixi run colab-status        # running machines and compute units left
pixi run colab-down          # stop the machine when you're done
```

- The first `colab` command asks you to sign in to Google. On the consent screen, tick every permission. If only your name and email are granted, the CLI stops with a "Scope has changed" error.
- `pixi run colab-up L4` picks another GPU. L4, A100, H100 and G4 need a paid Colab plan.
- Run `colab-up` once. A second run while the machine is up tries to create another machine with the same name. Check with `pixi run colab-status`.
- `colab-down` matters: a running machine keeps using your Colab quota.
- On Colab, the first output line names an NVIDIA GPU with `cuda` instead of `metal`, which confirms the puzzle ran there.

---

Previous: [Puzzle 07: 2D blocks](../p07/README.md)
