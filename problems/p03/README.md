# Puzzle 03: Guard

Add 10 to each element again, but launch more threads than the vector has elements. Any thread past the last element has to return without touching memory.

This is the first puzzle where the thread count and the data length are different numbers. The bounds check introduced here, `if i < size`, appears in almost every GPU kernel you will write.

## At a glance

| | |
|---|---|
| **Goal** | `output[i] = a[i] + 10`, only for `i < size` |
| **Data** | 1D vector, `SIZE = 4` floats |
| **Launch** | `grid_dim = 1` block, `block_dim = (8, 1)` threads |
| **Threads launched** | 8 (4 do work, 4 idle) |
| **New ideas** | bounds guard, passing the length as a kernel argument, 2-tuple `block_dim` |
| **File** | `p03.mojo` |

## Why there are extra threads

Real data rarely has a convenient length. GPU launches are rectangular: you pick a block size, and the hardware runs every thread in the block even if the data is smaller. Block sizes are usually chosen for the hardware (often a multiple of 32), not for the data. So you round **up** to cover all the data, and some threads at the end have nothing to do.

Here the vector has 4 elements and the block has 8 threads. Threads 4, 5, 6, and 7 have no element of their own.

Without a guard, thread 5 would compute `output[5] = a[5] + 10`. Both buffers hold only 4 floats, so that read and write land outside the allocation. On a GPU, that is **undefined behaviour**. It might crash, it might corrupt a neighbouring buffer, or it might seem to work and break later. A silent wrong answer is the worst outcome, and it is a common one.

## The kernel

```mojo
comptime SIZE = 4
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (8, 1)
comptime dtype = DType.float32

def add_10_guard(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var i = thread_idx.x
    # FILL ME IN (roughly 2 lines)
    if i < size:
        output[unsafe_offset=i] = a[unsafe_offset=i] + Scalar[dtype](10.0)
```

What changed from puzzle 01:

1. **`THREADS_PER_BLOCK = (8, 1)`.** The block size is now a tuple `(x, y)`. `(8, 1)` means 8 threads along x and 1 along y, so it is still a 1D row of threads. The y part matters starting in [puzzle 04](../p04/README.md).
2. **`size_dev: Int32`.** The pointer does not know how long its buffer is, so the length is passed as a separate argument. This 1.2.0 port passes scalar sizes as a fixed-width `Int32` and converts once at the top with `Int(size_dev)`, so the comparison below uses the same integer type as `thread_idx.x`.
3. **`if i < size:`.** This is the guard. Only threads whose index is a real element go on to read and write.
4. **`Scalar[dtype](10.0)`.** This spells out the constant's type. A bare `10.0` also works, as in puzzle 01.

## Which thread does what

```
 block_dim.x = 8
 thread     t0    t1    t2    t3  │  t4    t5    t6    t7
 i < 4 ?    yes   yes   yes   yes │  no    no    no    no
            │     │     │     │   │  (return without doing anything)
            ▼     ▼     ▼     ▼   │
 a       [ 0.0 , 1.0 , 2.0 , 3.0 ]│
 output  [10.0 ,11.0 ,12.0 ,13.0 ]│  ← end of the buffer
```

| Thread | `i` | `i < 4` | Action |
|---|---:|---|---|
| 0 | 0 | yes | `output[0] = 10` |
| 1 | 1 | yes | `output[1] = 11` |
| 2 | 2 | yes | `output[2] = 12` |
| 3 | 3 | yes | `output[3] = 13` |
| 4–7 | 4–7 | no | nothing |

Idle threads cost very little. They evaluate one comparison and finish.

## What `main` does

`main` is the same as puzzle 01 with one extra launch argument:

```mojo
ctx.enqueue_function[add_10_guard](
    out,
    a,
    Int32(SIZE),
    grid_dim=BLOCKS_PER_GRID,
    block_dim=THREADS_PER_BLOCK,
)
```

`Int32(SIZE)` is the length the kernel compares against. The buffers are still exactly `SIZE` floats long. The launch is bigger than the data, and the buffers are not. The final check loops over the 4 real elements only.

## Expected output

The first line names the device the kernel ran on: `metal` means the Apple GPU, `cuda` means an NVIDIA GPU (for example on Colab), and `cpu` would mean it ran on the CPU. The device name will differ on other machines.

```
Running on: Apple M3 Pro (metal, device 0)
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
Puzzle 03 complete ✅
```

The output looks identical to puzzle 01's. The difference is that 4 more threads ran and safely did nothing.

## Common mistakes

- **`<=` instead of `<`.** Valid indices are `0` through `size - 1`. `i <= size` lets thread 4 write one element past the end.
- **Guarding only the write.** If the read happens before the `if`, threads 4–7 still read out of bounds. Put the whole body inside the guard.
- **Comparing against the launch size.** `if i < THREADS_PER_BLOCK[0]` is always true. The guard has to compare against the data length.
- **Trusting that "it ran fine" means it is correct.** Out-of-bounds access often does not crash. Test with sizes that are not a multiple of the block size.

## Try it yourself

1. Delete the `if` (keep the assignment) and run it. It may still print the right answer, which is exactly why out-of-bounds bugs are dangerous. Put the guard back.
2. Change `SIZE` to 7 and leave the block at 8 threads. Only thread 7 should be idle.
3. Change `SIZE` to 10. Now 8 threads are not enough. Elements 8 and 9 stay 0 and the check fails. Raising the block size fixes that here, and [puzzle 06](../p06/README.md) shows the general fix: more blocks.

## Key takeaways

- Launch size and data size are separate numbers. Round the launch **up**, then guard.
- `if i < size` is the standard bounds check. Every thread that might be out of range must skip *all* memory access.
- The kernel has to receive the length as an argument, because a raw pointer does not carry one.
- Out-of-bounds GPU access is undefined behaviour, and it often fails silently.

## Run

This folder is its own Pixi project. It pins Mojo 1.2.0 and MAX 26.7.0 on the stable MAX channel.

```bash
cd problems/p03
pixi install
pixi run mojo p03.mojo
```

`pixi install` is once.

### Run on a Colab GPU

The project also has tasks that run the puzzle on a Google Colab GPU, using the [Google Colab CLI](https://developers.googleblog.com/en/introducing-the-google-colab-cli/) that `pixi install` sets up. Every puzzle folder uses the same Colab machine, named `mojo`, so you start it once and then run any puzzle on it.

```bash
pixi run colab-up            # start a T4 GPU machine, install Mojo 1.2.0 + MAX 26.7.0 on it
pixi run colab-run           # upload p03.mojo and run it there
pixi run colab-status        # running machines and compute units left
pixi run colab-down          # stop the machine when you're done
```

- The first `colab` command asks you to sign in to Google. On the consent screen, tick every permission. If only your name and email are granted, the CLI stops with a "Scope has changed" error.
- `pixi run colab-up L4` picks another GPU. L4, A100, H100 and G4 need a paid Colab plan.
- Run `colab-up` once. A second run while the machine is up tries to create another machine with the same name. Check with `pixi run colab-status`.
- `colab-down` matters: a running machine keeps using your Colab quota.
- On Colab, the first output line names an NVIDIA GPU with `cuda` instead of `metal`, which confirms the puzzle ran there.

---

Previous: [Puzzle 02: Zip](../p02/README.md) · Next: [Puzzle 04: 2D map](../p04/README.md)
