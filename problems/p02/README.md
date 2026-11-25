# Puzzle 02: Zip

Add two vectors into a third, element by element.

The launch is the same shape as [puzzle 01](../p01/README.md). What is new is a kernel with two inputs. The same thread index selects one element from each input and one from the output, which "zips" the buffers together.

## At a glance

| | |
|---|---|
| **Goal** | `output[i] = a[i] + b[i]` |
| **Data** | three 1D vectors, `SIZE = 4` floats each |
| **Launch** | `grid_dim = 1` block, `block_dim = 4` threads |
| **Threads launched** | 4 (one per element, none idle) |
| **New ideas** | more than one input buffer, one index shared across buffers |
| **File** | `p02.mojo` |

## The idea

In Python this would be:

```python
output = [x + y for x, y in zip(a, b)]
```

On the GPU, the `zip` disappears. Thread `i` does not need to search for its pair: element `i` of `a`, element `i` of `b`, and element `i` of `output` all sit at the same offset in their own buffers. One index lines all three up.

This only works because the three buffers have the **same length and the same layout**. When the shapes differ, as in [puzzle 05](../p05/README.md), each input needs its own indexing rule.

## The kernel

```mojo
comptime SIZE = 4
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = SIZE
comptime dtype = DType.float32

def add(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
):
    var i = thread_idx.x
    # FILL ME IN (roughly 1 line)
    output[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]
```

- `output`, `a`, and `b` are three raw pointers into three separate device buffers.
- `i = thread_idx.x` is the thread's position in its block. With one block, it is also the element index.
- The single line under `# FILL ME IN` is the solution: two reads, one add, one write.

The three pointers all use `MutAnyOrigin`, even though the kernel never writes to `a` or `b`. Raw pointers here do not separate read-only inputs from writable outputs. Puzzle 05 is the first kernel that marks its inputs read-only, with `mut=False` on a `TileTensor`.

## Which thread does what

```
 thread      t0      t1      t2      t3
              │       │       │       │
 a        [  0.0  ,  1.0  ,  2.0  ,  3.0  ]
              +       +       +       +
 b        [  0.0  ,  1.0  ,  2.0  ,  3.0  ]
              =       =       =       =
 output   [  0.0  ,  2.0  ,  4.0  ,  6.0  ]
```

| Thread | `a[i]` | `b[i]` | `output[i]` |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 1 | 2 |
| 2 | 2 | 2 | 4 |
| 3 | 3 | 3 | 6 |

Each thread touches exactly three memory locations, and no two threads touch the same output location. There is no shared state, so no coordination is needed.

## What `main` does

The structure is the same as puzzle 01, with one more buffer and one difference in when `expected` is built.

| Step | Code | What happens |
|---|---|---|
| 1 | `enqueue_create_buffer` ×3 | Allocates `out`, `a`, and `b` on the device and queues a zero fill for each. |
| 2 | `enqueue_create_host_buffer` | Allocates `expected` on the host and queues a zero fill. |
| 3 | `with a.map_to_host() as a_host, b.map_to_host() as b_host:` | Maps both inputs at once. The loop writes `i` into both, and also computes `expected[i] = a_host[i] + b_host[i]`. |
| 4 | `ctx.enqueue_function[add](out, a, b, ...)` | Launches the kernel with three buffer arguments, in the same order as the kernel's parameters. |
| 5 | `ctx.synchronize()` | Waits for the kernel. |
| 6 | `with out.map_to_host() as out_host:` | Prints both buffers and calls `assert_equal` on every element. |

Here `expected` is filled *inside* the `map_to_host` block, before the launch, while puzzle 01 filled it after `synchronize()`. Both are safe. Entering `map_to_host` waits for the queued work, including the zero fill on `expected`, so the host writes that follow are not overwritten.

The expected answer comes from the same host values the GPU receives. The check compares the GPU's `a + b` with the CPU's `a + b`, not with a hand-written list.

## Expected output

The first line names the device the kernel ran on: `metal` means the Apple GPU, `cuda` means an NVIDIA GPU (for example on Colab), and `cpu` would mean it ran on the CPU. The device name will differ on other machines.

```
Running on: Apple M3 Pro (metal, device 0)
out: HostBuffer([0.0, 2.0, 4.0, 6.0])
expected: HostBuffer([0.0, 2.0, 4.0, 6.0])
Puzzle 02 complete ✅
```

## Common mistakes

- **Argument order.** `enqueue_function[add](out, a, b, ...)` binds by position. Swapping `out` and `a` makes the kernel write into the input buffer and leaves `out` at zero.
- **Inputs that are equal by accident.** `a` and `b` hold the same values here, so `a + a` would also pass. Make `b` different (see below) to be sure the kernel really reads `b`.
- **Different lengths.** If `b` were shorter than `a`, thread `i` would still read `b[i]` and run past the end. The shared index assumes equal lengths.

## Try it yourself

1. Fill `b` with `Scalar[dtype](i * 10)` instead of `i`. The expected answer becomes `[0, 11, 22, 33]`, and a kernel that wrongly reads `a` twice will now fail.
2. Change the kernel to `a[i] * b[i]` and update the `expected` line to match.
3. Add a fourth buffer `c` and compute `a + b + c`. You need a new parameter, a new allocation, a new `map_to_host`, and a new launch argument.

## Key takeaways

- One thread index can address several buffers at once, as long as they share the same length and layout.
- Element-wise operations (add, multiply, compare) all have this shape: a few reads at index `i`, one write at index `i`.
- Raw pointers carry no shape or read-only information. Later puzzles move that information into the type.

## Run

This folder is its own Pixi project. It pins Mojo 1.2.0 and MAX 26.7.0 on the stable MAX channel.

```bash
cd problems/p02
pixi install
pixi run mojo p02.mojo
```

`pixi install` is once.

### Run on a Colab GPU

The project also has tasks that run the puzzle on a Google Colab GPU, using the [Google Colab CLI](https://developers.googleblog.com/en/introducing-the-google-colab-cli/) that `pixi install` sets up. Every puzzle folder uses the same Colab machine, named `mojo`, so you start it once and then run any puzzle on it.

```bash
pixi run colab-up            # start a T4 GPU machine, install Mojo 1.2.0 + MAX 26.7.0 on it
pixi run colab-run           # upload p02.mojo and run it there
pixi run colab-status        # running machines and compute units left
pixi run colab-down          # stop the machine when you're done
```

- The first `colab` command asks you to sign in to Google. On the consent screen, tick every permission. If only your name and email are granted, the CLI stops with a "Scope has changed" error.
- `pixi run colab-up L4` picks another GPU. L4, A100, H100 and G4 need a paid Colab plan.
- Run `colab-up` once. A second run while the machine is up tries to create another machine with the same name. Check with `pixi run colab-status`.
- `colab-down` matters: a running machine keeps using your Colab quota.
- On Colab, the first output line names an NVIDIA GPU with `cuda` instead of `metal`, which confirms the puzzle ran there.

---

Previous: [Puzzle 01: Map](../p01/README.md) · Next: [Puzzle 03: Guard](../p03/README.md)
