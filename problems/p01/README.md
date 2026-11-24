# Puzzle 01: Map

Add 10 to every element of a vector, with one GPU thread responsible for one element.

This is the smallest kernel in the set: one block, no bounds check, and a flat pointer on each side. It also introduces the host-side pattern that every later puzzle reuses: allocate device buffers, fill them, launch a kernel, wait, copy the result back, and check it.

## At a glance

| | |
|---|---|
| **Goal** | `output[i] = a[i] + 10` |
| **Data** | 1D vector, `SIZE = 4` floats (`float32`) |
| **Launch** | `grid_dim = 1` block, `block_dim = 4` threads |
| **Threads launched** | 4 (one per element, none idle) |
| **New ideas** | kernel, thread, `thread_idx`, `DeviceContext`, device buffers |
| **File** | `p01.mojo` |

## Vocabulary

GPU code is written from the point of view of a single thread, and the hardware runs many copies of it at once. A few words come up in every puzzle:

| Term | Meaning |
|---|---|
| **Host** | The CPU and its memory. `main` runs here. |
| **Device** | The GPU and its memory. The kernel runs here. |
| **Kernel** | A function that runs on the device, once per thread. Here it is `add_10`. |
| **Thread** | One running copy of the kernel. Each has its own index. |
| **Block** | A group of threads launched together. `block_dim` is its size. |
| **Grid** | All the blocks of one launch. `grid_dim` is the number of blocks. |
| **`thread_idx.x`** | This thread's position inside its block, starting at 0. |

In this puzzle there is only one block, so `thread_idx.x` is also the thread's position in the whole launch.

## The kernel

```mojo
comptime SIZE = 4
comptime BLOCK_PER_GRID = 1
comptime THREADS_PER_BLOCK = SIZE
comptime dtype = DType.float32

def add_10(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
):
    var i = thread_idx.x
    output[unsafe_offset=i] = a[unsafe_offset=i] + 10.0
```

Line by line:

1. `comptime` values are compile-time constants. The launch shape and the element type are fixed before the program runs.
2. Both arguments are `Pointer[Scalar[dtype], MutAnyOrigin]`: a raw pointer to `float32` values in device memory. The kernel gets no length, no shape, and no bounds information. It only gets an address.
3. `var i = thread_idx.x` asks "which thread am I?" Every thread runs this same line and gets a different answer: 0, 1, 2, or 3.
4. `output[unsafe_offset=i] = a[unsafe_offset=i] + 10.0` reads one element, adds 10, and writes one element. Mojo 1.2.0 spells pointer indexing as `unsafe_offset`, so the call site shows that nothing checks the index.

There is no loop. In CPU code you would write `for i in range(SIZE)`. On the GPU, the loop is replaced by launching `SIZE` threads, and each thread does one iteration.

## Which thread does what

```
               one block, block_dim.x = 4
            ┌────────┬────────┬────────┬────────┐
 thread     │ t0     │ t1     │ t2     │ t3     │
            └───┬────┴───┬────┴───┬────┴───┬────┘
                │ i=0    │ i=1    │ i=2    │ i=3
                ▼        ▼        ▼        ▼
 a        [    0.0  ,   1.0  ,   2.0  ,   3.0   ]
               +10      +10      +10      +10
 output   [   10.0  ,  11.0  ,  12.0  ,  13.0   ]
```

| Thread | `i` | Reads | Writes |
|---|---:|---|---|
| 0 | 0 | `a[0] = 0` | `output[0] = 10` |
| 1 | 1 | `a[1] = 1` | `output[1] = 11` |
| 2 | 2 | `a[2] = 2` | `output[2] = 12` |
| 3 | 3 | `a[3] = 3` | `output[3] = 13` |

The four threads do not talk to each other and do not depend on each other's results, so the order they run in does not matter. That independence is what makes this a *map*.

There is no `if` around the write. That is safe only because the thread count and the vector length are the same number (`THREADS_PER_BLOCK = SIZE`). [Puzzle 03](../p03/README.md) is the first program that launches extra threads and has to skip them.

## What `main` does

The host code is longer than the kernel. Almost every later puzzle follows these same steps.

| Step | Code | What happens |
|---|---|---|
| 1 | `with DeviceContext() as ctx:` | Opens a handle to the GPU. Work is queued on it in order. |
| 2 | `ctx.enqueue_create_buffer[dtype](SIZE)` | Allocates `out` and `a` in device memory. |
| 3 | `out.enqueue_fill(0)`, `a.enqueue_fill(0)` | Queues a fill so the buffers start at zero, not garbage. |
| 4 | `with a.map_to_host() as a_host:` | Gives the host a view of `a`. The loop writes `0, 1, 2, 3`, and the data goes back to the device when the `with` block ends. |
| 5 | `ctx.enqueue_function[add_10](out, a, grid_dim=..., block_dim=...)` | Queues the kernel launch. The device buffers are passed where the kernel expects pointers. |
| 6 | `ctx.enqueue_create_host_buffer[dtype](SIZE)` | Allocates `expected` in host memory and queues a zero fill. |
| 7 | `ctx.synchronize()` | Blocks until everything queued so far has finished, including the kernel. |
| 8 | `expected[i] = i + 10` | Builds the reference answer on the CPU. |
| 9 | `with out.map_to_host() as out_host:` | Reads the GPU result back, prints both buffers, and calls `assert_equal` on every element. |

The word *enqueue* matters. `enqueue_function` returns right away, before the kernel has run. The result is only safe to read after `ctx.synchronize()` (or a `map_to_host`, which also waits). `expected` is written after `synchronize()` for the same reason: its zero fill was queued, and writing to it before the fill has run could be overwritten.

## Expected output

```
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
Puzzle 01 complete ✅
```

If any element differs, `assert_equal` raises and the program stops before the final line.

## Common mistakes

- **Reading the result before synchronizing.** The launch is asynchronous. Without `ctx.synchronize()`, the host may see the zeros from `enqueue_fill`.
- **Writing a loop inside the kernel.** `for j in range(SIZE)` inside `add_10` would make every thread do all four elements, so each element gets written four times. The answer happens to come out the same here, but the work is 4× larger and stops being parallel.
- **Launching fewer threads than elements.** With `THREADS_PER_BLOCK = 2`, only `output[0]` and `output[1]` are written. The rest stay 0 and the check fails at index 2.
- **Launching more threads than elements without a guard.** Threads past the end would read and write outside the buffers. That is undefined behaviour, and it may not crash. Puzzle 03 fixes this.

## Try it yourself

1. Change `SIZE` to 8. `THREADS_PER_BLOCK` follows it, so the program still passes, now with 8 threads.
2. Set `THREADS_PER_BLOCK = 2` and keep `SIZE = 4`. Read the printed `out` and see which elements were never written.
3. Change `+ 10.0` to `* 2.0` and update the `expected` loop to match.
4. Add `print("thread", i)` inside the kernel. The lines may not come out in thread order, which shows that the threads run independently.

## Key takeaways

- A kernel describes the work of **one thread**. Parallelism comes from launching many threads, not from a loop.
- `thread_idx.x` is how a thread finds its element.
- The host allocates, fills, launches, synchronizes, and checks. GPU work is **queued**, so you must wait before reading results.
- A kernel with no bounds check is correct only when the thread count exactly matches the data.

## Run

This folder is its own Pixi project. It pins Mojo 1.2.0 and MAX 26.7.0 on the stable MAX channel. `max` has to be installed here: a bare `mojo` binary does not provide `max.gpu`.

```bash
cd problems/p01
pixi install
pixi run mojo p01.mojo
```

`pixi install` is once.

---

Next: [Puzzle 02: Zip](../p02/README.md)
