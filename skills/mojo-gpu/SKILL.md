---
name: mojo-gpu
---

# Mojo GPU

New GPU code uses the MAX accelerator library. There is no CUDA syntax: no `__global__`, `<<<>>>`, `threadIdx`, or `__syncthreads__`. Host Mojo follows `mojo-manual`. Do not write `fn`, `from gpu import`, `LayoutTensor`, `alias`, `@parameter`, or `enqueue_function_checked`.

Two sources, and they do not share one `TileTensor` signature:

- Intro and fundamentals: <https://max.modular.com/gpu/intro-tutorial.md>, <https://max.modular.com/gpu/fundamentals.md>, <https://max.modular.com/gpu/block-and-warp.md>. `TileTensor[dtype, layout, origin]` with no `Engine`. `row_major` from `layout.layout`. `stack_allocation` from `layout`.
- The puzzle book: <https://puzzles.modular.com>, matching `github.com/modular/mojo-gpu-puzzles` branch `stable`. `main` is nightly and can disagree with the book. From puzzle 5, `TileTensor[mut=..., dtype, layout, origin, Engine=Engine]` and `where (Engine.element_size == 1)`. `row_major` from `layout.tile_layout`. `stack_allocation` from `layout.tile_tensor`.

Match the file you are editing. On a puzzle, open that puzzle's page and change only the `FILL ME IN` region. Keep its signature, imports, origins, and names.

## What runs where

| Piece | Where it runs | Typical type |
|---|---|---|
| `main`, allocation, copies | Host CPU | `DeviceContext`, `DeviceBuffer`, `HostBuffer` |
| Kernel body | GPU | `Pointer` or `TileTensor` |
| `thread_idx`, `block_idx`, `global_idx` | GPU only | `Int` |
| `has_accelerator()` | Host, compile time | `Bool` |

A kernel is a plain `def`. It does not raise and does not return a value. Write results through an argument. `Int` and `UInt` are not `DevicePassable`. Pass `Int32` or `Int64`, then `var size = Int(size_dev)` inside the kernel. `Int` inside the kernel body is fine.

`has_accelerator()` from `std.sys` is the host check. Vendor checks are `ctx.target.is_nvidia_gpu()`, `is_amd_gpu()`, and `is_apple_gpu()`. Host predicates in `std.sys.info` describe the CPU, so a GPU path guarded by them is compiled out. `DeviceContext()` raises when that device is missing. Keep the GPU branch under `comptime if has_accelerator():`.

## Imports

```mojo
from std.memory import Pointer
from std.sys import has_accelerator, simd_width_of, align_of
from std.math import ceildiv
from max.gpu.host import DeviceContext
from max.gpu.host.compile import get_gpu_target
from max.gpu import thread_idx, block_idx, block_dim, grid_dim, global_idx, lane_id
from max.gpu.sync import barrier
from max.gpu.memory import async_copy_wait_all
from max.gpu.primitives.warp import sum as warp_sum, WARP_SIZE, shuffle_down, broadcast, shuffle_xor, prefix_sum
from max.gpu.primitives import block
from max.algorithm.functional import elementwise
from layout import TileTensor, TensorEngine, Coord
from layout.tile_io import copy_dram_to_sram_async
from layout.tensor_core import TensorCore
```

Import only what the file needs. `Pointer` is `std.memory.Pointer`. Fundamentals code adds `from layout import stack_allocation` and `from layout.layout import row_major`. Puzzle code adds `from layout.tile_layout import row_major` and `from layout.tile_tensor import stack_allocation`.

| Stale | Current |
|---|---|
| `from gpu import …` or `from std.gpu import …` | `from max.gpu import …` |
| `ptr[i]` on a `Pointer` | `ptr[unsafe_offset=i]` |
| `__syncthreads()` | `barrier()` |
| `threadIdx.x` | `thread_idx.x` |
| `std.sys.info.is_nvidia_gpu()` on the host | `ctx.target.is_nvidia_gpu()` |

## Host program

`DeviceBuffer` is the host name of a device allocation. The kernel receives it as `Pointer`. Putting `DeviceBuffer` in the kernel signature is an error.

```mojo
from std.math import iota
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from max.gpu import global_idx

comptime n = 1024
comptime block = 256

def saxpy(
    out: Pointer[Float32, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    size: Int32,
    a: Float32,
):
    var i = global_idx.x
    if i < Int(size):
        out[unsafe_offset=i] = a * x[unsafe_offset=i] + y[unsafe_offset=i]

def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
    else:
        var ctx = DeviceContext()
        var x_host = ctx.enqueue_create_host_buffer[.float32](n)
        var y_host = ctx.enqueue_create_host_buffer[.float32](n)
        ctx.synchronize()
        iota(x_host.as_span())
        iota(y_host.as_span())

        var x_dev = ctx.enqueue_create_buffer[.float32](n)
        var y_dev = ctx.enqueue_create_buffer[.float32](n)
        var out_dev = ctx.enqueue_create_buffer[.float32](n)
        ctx.enqueue_copy(src_buf=x_host, dst_buf=x_dev)
        ctx.enqueue_copy(src_buf=y_host, dst_buf=y_dev)

        ctx.enqueue_function[saxpy](
            out_dev, x_dev, y_dev, Int32(n), Float32(2.0),
            grid_dim=ceildiv(n, block),
            block_dim=block,
        )
        ctx.enqueue_copy(src_buf=out_dev, dst_buf=x_host)
        ctx.synchronize()
        print(x_host)
```

`enqueue_function[kernel](...)` compiles and launches when the kernel's parameters are already bound. If parameters remain, bind them first (`comptime k = kernel[dtype]`) and launch `enqueue_function[k]`. An `Engine` kernel launches as `enqueue_function[kernel[out_tensor.Engine]]`. `compile_function[kernel]()` returns a handle for the non-parameterized `enqueue_function(handle, args, ...)`.

The stream is ordered and asynchronous. `enqueue_*` returns before the GPU finishes. Call `synchronize()` before the host reads memory the GPU just wrote. `map_to_host` synchronizes for you. `device.enqueue_fill(0.0)` fills a device buffer. `DeviceContext(device_id=0)` takes `api="cuda"`, `"hip"`, or `"metal"`. `DeviceContext.number_of_devices()` counts devices. Puzzle host code usually writes `with DeviceContext() as ctx`. New fundamentals code more often writes `var ctx = DeviceContext()`.

## Launch shape

`grid_dim` is the number of blocks. `block_dim` is the threads in one block. Each accepts an `Int` or `(x,)`, `(x, y)`, `(x, y, z)`. A missing axis is 1. Row comes from `y`. Column comes from `x`.

| Name | Meaning |
|---|---|
| `thread_idx` | Index inside the block |
| `block_idx` | Block index inside the grid |
| `block_dim` | Threads per block |
| `grid_dim` | Blocks in the grid |
| `global_idx` | `block_dim * block_idx + thread_idx`, per axis |

```mojo
var row = block_idx.y * block_dim.y + thread_idx.y
var col = block_idx.x * block_dim.x + thread_idx.x
if row < Int(rows) and col < Int(cols):
    out[row, col] = a[row, col]
```

A grid larger than the data is normal. Guard every global read and write. On Apple GPUs, `print` inside a kernel accepts at most one string literal. Print numbers from the kernel and headers from the host.

`WARP_SIZE` is 32 on NVIDIA, RDNA, and Apple, and 64 on AMD CDNA. A block size that does not divide the warp still runs, but the tail warp occupies a full warp of resources.

## Pointer kernels

Puzzles 1–4 (raw) and 6 take `Pointer[Scalar[dtype], MutAnyOrigin]` and index with `unsafe_offset`. Puzzle 1 uses one block with `THREADS_PER_BLOCK = SIZE`, so no guard. Puzzle 3 launches more threads than elements, so `if i < size`. Puzzle 6 is the grid index `i = block_dim.x * block_idx.x + thread_idx.x` plus that guard.

## TileTensor

Indexing yields a SIMD value. `a[i][0]` is the scalar lane. When two layouts do not unify, `rebind[Scalar[dtype]](a[i])` before scalar arithmetic. `load[width]` and `store[width]` take a `Coord`. `.tile[rows, cols](tile_row, tile_col)` is a block's view. `.vectorize[1, 4]()` widens the inner dimension. A layout that is a parameter needs `comptime assert tensor.flat_rank == N` before a subscript. Re-assert after `.tile()`, `.vectorize()`, or `.distribute()`.

Fundamentals form, no `Engine`:

```mojo
comptime layout = row_major[64, 64]()

def kernel(
    a: TileTensor[dtype, type_of(layout), ImmutAnyOrigin],
    c: TileTensor[dtype, type_of(layout), MutAnyOrigin],
):
    var shared = stack_allocation[dtype, address_space=.SHARED](row_major[16, 16]())
    shared[thread_idx.y, thread_idx.x] = a[global_row, global_col]
    barrier()
```

Puzzle 4's `p04_tile_tensor.mojo` is the early puzzle form: no `Engine`, both tensors `mut=True`, `output[row, col] = a[row, col] + 10.0`. Do not add `Engine` there.

From puzzle 5 the output is `mut=True` with `MutAnyOrigin`, inputs are `mut=False`, and the input origin is whatever the file wrote (`ImmutAnyOrigin` or `MutAnyOrigin`). Layouts are `comptime layout = row_major[SIZE, SIZE]()` and `comptime LayoutType = type_of(layout)`.

```mojo
def broadcast_add[Engine: TensorEngine](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin, Engine=Engine],
    a: TileTensor[mut=False, dtype, ALayout, ImmutAnyOrigin, Engine=Engine],
    b: TileTensor[mut=False, dtype, BLayout, ImmutAnyOrigin, Engine=Engine],
    size_dev: Int32,
) where (Engine.element_size == 1):
    var size = Int(size_dev)
    var row = thread_idx.y
    var col = thread_idx.x
    if row < size and col < size:
        output[row, col] = a[0, col] + b[row, 0]
```

## Shared memory and barriers

```mojo
var shared = stack_allocation[dtype=dtype, address_space=.SHARED](row_major[TPB]())
if global_i < size:
    shared[local_i] = rebind[Scalar[dtype]](a[global_i])
barrier()
```

`stack_allocation[dtype, .SHARED](layout)` is the same call. Without `address_space=.SHARED` it is per-thread local memory. Shared memory is per block and its shape is a compile-time constant. Shared slots use the local index, and only after that load was in range.

`barrier()` is a block barrier and a memory fence. Every thread in the block must reach it, so it stays outside a per-thread `if`. Call it after the cooperative load and again before the next tile overwrites shared memory. It does not reach other blocks. A result that depends on another block is a second kernel on the same stream, enqueued after the first, with no host synchronize between the launches. Warp and block collectives already synchronize their participants. Do not put `barrier()` or `syncwarp()` in front of `warp_sum`, a shuffle, or `block.sum`. `syncwarp()` is a real instruction on NVIDIA and a no-op on AMD.

## Puzzle map

Clone `stable`, then `pixi run p01`. Add `-e amd` or `-e apple` on those GPUs. `pixi run mojo -I . problems/p01/p01.mojo` and `pixi run tests p01` also work. `uv run poe pXX` works where the puzzle says so.

`PuzzleMemory` puts the output in a larger buffer with NaN margins. `❌ Write detected outside of output buffer` means an index ran past the output, usually a missing guard on the last block. Puzzles 9, 10, 17–20, and 30–32 do not print that check. Puzzles 30–32 have no reference solution.

| Puzzles | Work | Where it runs |
|---|---|---|
| 1–8 | Map, zip, guards, 2D map, broadcast, blocks, 2D blocks, shared memory | NVIDIA, AMD, Apple |
| 9–10 | `mojo debug` after `pixi run setup-cuda-gdb`. Puzzle 10 is `compute-sanitizer` memcheck and racecheck | NVIDIA |
| 11–15 | Pool, dot product, 1D convolution, prefix sum, axis sum | all three |
| 16 | Matmul: naive, shared-memory block, tiled, then async copy. Needs compute capability 8.0 | all three, not a T4 |
| 17–19 | MAX custom ops: conv1d, softmax, attention | all three |
| 20–22 | The same ops from PyTorch. 22 has a backward kernel | NVIDIA, AMD |
| 23 | `elementwise[..., target="gpu"]`: plain, tiled, vectorized, vectorize-inside-tile | all three |
| 24–26 | `warp_sum`, `shuffle_down` / `broadcast`, `shuffle_xor` / `prefix_sum` | all three |
| 27 | `block.sum`, `block.prefix_sum`, `block.broadcast` | all three |
| 28 | Async copy overlapped with compute. Needs 8.0 | all three, not a T4 |
| 29 | `mbarrier_init`, `mbarrier_arrive`, `mbarrier_test_wait`. Needs 8.0 | NVIDIA |
| 30–32 | Nsight (`ncu`, `nsys`), occupancy, shared-memory bank conflicts. No solution file | NVIDIA |
| 33 | `TensorCore`, MMA tile 16×8×8. Needs 8.0 | NVIDIA |
| 34 | Cluster sync. Needs SM90 | NVIDIA Hopper or newer |
| 35 | `load` / `store` alignment | all three |

Puzzle 12 is a one-block tree reduction and thread 0 stores the scalar. Puzzle 13's block-boundary kernel loads a halo so the window can cross the edge. Puzzle 14's full scan is two kernels: a local scan that records the block total, then a kernel that adds the preceding blocks. Puzzle 15 is one block per row (`block_idx.y`).

Puzzle 16's idiomatic matmul:

```mojo
var out_tile = output.tile[TPB, TPB](block_idx.y, block_idx.x)
var acc: output.ElementType = 0
comptime for idx in range(size // TPB):
    copy_dram_to_sram_async[thread_layout=load_a_layout, num_threads=NUM_THREADS](a_shared, a.tile[TPB, TPB](block_idx.y, Int(idx)))
    copy_dram_to_sram_async[thread_layout=load_b_layout, num_threads=NUM_THREADS](b_shared, b.tile[TPB, TPB](Int(idx), block_idx.x))
    async_copy_wait_all()
    barrier()
    comptime for k in range(TPB):
        acc += a_shared[local_row, k] * b_shared[k, local_col]
    barrier()
```

Both tiles stay in the same orientation. `async_copy_wait_all` waits for the copies. The following `barrier()` publishes those shared writes to the rest of the block.

## Custom ops

Puzzles 17–22 register a struct. `InputTensor` and `OutputTensor` come from `extensibility`.

```mojo
@extensibility.register("conv1d")
struct Conv1DCustomOp:
    @staticmethod
    def execute[target: StaticString, input_size: Int, conv_size: Int, dtype: DType = .float32](
        output: OutputTensor[dtype=dtype, rank=1, static_spec=_],
        input: InputTensor[dtype=dtype, rank=output.rank, static_spec=_],
        kernel: InputTensor[dtype=dtype, rank=output.rank, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        var output_tensor = output.to_tile_tensor().as_unsafe_any_origin()
        var input_tensor = input.to_tile_tensor().as_unsafe_any_origin()
        var kernel_tensor = kernel.to_tile_tensor().as_unsafe_any_origin()
        comptime if target == "gpu":
            comptime kernel = conv1d_kernel[
                input_size, conv_size,
                output_tensor.LayoutType, input_tensor.LayoutType, kernel_tensor.LayoutType,
                output_tensor.Engine,
            ]
            ctx.enqueue_function[kernel](
                output_tensor, input_tensor, kernel_tensor,
                grid_dim=BLOCKS_PER_GRID, block_dim=(TPB, 1),
            )
```

Puzzle 18 is softmax (max, subtract, `exp` from `std.math`, sum, normalize). Puzzle 19 is attention: tiled matmul, transpose through shared memory, softmax, second matmul. Puzzle 22 is fused layer-norm plus linear, with a separate backward op under its own name. The CPU branch of `execute` must still return the right answer.

## Functional launch, warps, and blocks

Puzzle 23 passes a closure to `elementwise` from `max.algorithm.functional`. `SIMD_WIDTH` is `simd_width_of[dtype, target=get_gpu_target()]()`. The closure capture is `{var}` when it writes captured state. The function `raises` because it receives `DeviceContext`.

```mojo
def add[simd_width: Int, alignment: Int = 1](indices: Coord) {var} -> None:
    var idx = Int(indices[0].value())
    var ret = a.load[width=simd_width](Coord(idx)) + b.load[width=simd_width](Coord(idx))
    output.store[simd_width](Coord(idx), ret)

elementwise[simd_width=SIMD_WIDTH, target="gpu"](add, Coord(size), ctx)
```

The four variants are one vector per thread, one tile per thread (`elementwise` over `ceildiv(size, tile_size)` with `simd_width=1`), a manually vectorized tile, and `vectorize` inside the tile.

Puzzle 35 states alignment. `align_of[SIMD[dtype, SIMD_WIDTH]]()` is the wide-load alignment, and `load[width=SIMD_WIDTH]` uses it by default. Passing `alignment=align_of[dtype]()` still computes the right values and makes the compiler emit scalar loads. Guard with `base + SIMD_WIDTH <= size`.

`warp_sum` returns the total on every lane. Only `lane_id() == 0` writes: `output.store[1](Coord(global_i // WARP_SIZE), total)`. `shuffle_down(value, offset)` reads a higher lane. `broadcast` publishes one lane. An offset that leaves the warp is undefined, so the last lane writes a defined fallback. `shuffle_xor` is the butterfly. `prefix_sum` is the inclusive warp scan. `shuffle_up` and `shuffle_idx` are the other neighbor exchanges. A full mask still requires every lane to reach the shuffle.

`block.sum[block_size=tpb, broadcast=False](val=Float32(partial))` reduces the block. `broadcast=False` leaves the total on the receiving thread. Every thread still calls it, and thread 0 writes `output[0] = total[0]`. `block.prefix_sum` produces write offsets. `block.broadcast` shares one thread's value. A block collective is the one to use when the block spans more than one warp. Across blocks, use shared memory plus `barrier()`, then a second kernel if another block must see the result.

## Async copy, profiling, tensor cores, clusters

Puzzle 28 overlaps `copy_dram_to_sram_async` with compute and finishes with `async_copy_wait_all`. The convolution tile includes the halo (`KERNEL_SIZE // 2` on each side).

Puzzle 29's barriers are `stack_allocation[dtype=DType.uint64, address_space=.SHARED](row_major[1]())`. Thread 0 calls `mbarrier_init(barrier.ptr, thread_count)`, then a `barrier()` makes the object visible before any `mbarrier_arrive`. A stage waits with `while not mbarrier_test_wait(barrier.ptr, token)`. `mbarrier_init`, `mbarrier_arrive`, and `mbarrier_test_wait` come from `max.gpu.sync`.

Puzzle 30 is the Nsight workflow on the puzzle page. Puzzle 31 is occupancy: registers and shared memory decide how many blocks stay resident. Puzzle 32 is bank conflicts. Threads in a warp that hit the same bank serialize. Change the inner stride, including padding, to the conflict-free pattern the page shows.

Puzzle 33: `TensorCore[dtype, dtype, Index(MMA_M, MMA_N, MMA_K)]()` with a 16×8×8 MMA tile. `BM`, `BN`, `BK` tile the block. `WM`, `WN` tile the warp. Copies into shared memory are async and vectorized. The accumulator is `stack_allocation[dtype, .LOCAL]`. An inactive warp skips the MMA.

Puzzle 34 uses `block_rank_in_cluster`, `cluster_sync`, `cluster_arrive`, `cluster_wait`, and `elect_one_sync` from `max.gpu.primitives.cluster`. `cluster_sync` waits for the whole cluster. Arrive and wait split that. `elect_one_sync` picks one thread for the cluster-wide side effect. Launch with the cluster dimension the file declares (`Dim` from `max.gpu.host`).

## Toolchain

```bash
pixi run mojo --version
mojo --print-supported-accelerators
mojo run --target-accelerator sm_90 kernel.mojo
mojo debug --cuda-gdb --break-on-launch kernel.mojo
git clone --branch stable https://github.com/modular/mojo-gpu-puzzles
pixi run gpu-specs
```

The `max` package supplies `max.gpu`. The `mojo` package alone does not. A kernel fails at compile time without a supported GPU toolchain. LLDB does not see GPU code. `--cuda-gdb` is NVIDIA-only.

Puzzles 16, 28, 29, and 33 need compute capability 8.0, so a T4 cannot run them. Puzzle 34 needs SM90. Puzzles 9, 10, and 29–34 do not run on Apple GPUs. Puzzles 20–22 do not run on Apple GPUs either.
