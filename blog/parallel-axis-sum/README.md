*Updated November 2026 for Mojo 1.2.0. The original post (31 Aug 2025) used GPU APIs that have since moved or been renamed. Both programs below have been compiled and run with Mojo 1.2.0, on an Apple M3 Pro GPU. A summary of what changed is at the end.*

When working with multidimensional arrays on GPUs, one of the most common operations you'll encounter is axis summation: reducing a tensor along a specific dimension. Today we'll explore two implementations that show the power of shared memory parallel reduction, covering both axis=0 (column-wise) and axis=1 (row-wise) summation.

## What is Axis Summation?

Before diving into the GPU implementation, let's establish what we mean by axis summation. Imagine you have a 2D matrix like this:

Matrix (8×16): Each element = row × 10 + column

![8x16 matrix where each element is row × 10 + column](https://bear-images.sfo2.cdn.digitaloceanspaces.com/kaka/svgviewer-output-2-4.webp)

Axis operations:

- **Axis=0 sum (↓):** Sum down each column → 16 results

  `[280, 288, 296, 304, ..., 384, 392, 400]`

- **Axis=1 sum (→):** Sum across each row → 8 results

  `[120, 280, 440, 600, 760, 920, 1080, 1240]`

**Axis=0 sum (column-wise):** We sum down each column, collapsing the rows but keeping the columns. The result has 16 elements, one sum per column.

**Axis=1 sum (row-wise):** We sum across each row, collapsing the columns but keeping the rows. The result has 8 elements, one sum per row.

The challenge in GPU programming is to perform these reductions efficiently while making use of the GPU's parallelism.

## The Power of Shared Memory Reduction

Both implementations use a technique called **shared memory parallel reduction**. This approach is crucial for GPU efficiency because:

- **Shared memory is fast:** it's much faster than global memory.
- **Parallel tree reduction:** threads work together, so the reduction takes O(log n) steps instead of O(n).
- **It minimizes memory traffic:** data is loaded from global memory once, then processed entirely in fast shared memory.

Let's see how this works by walking through each implementation.

## Impl. 1: Axis=0 Sum (Column-wise Reduction)

The first implementation tackles what's typically the harder case: summing along rows to produce column sums.

### The Challenge: Strided Memory Access

```mojo
# PHASE 1: Load data into shared memory
# Each thread loads one element from its assigned row in this column
# NOTE: This creates a strided memory access pattern across threads
shared[tid] = input_tensor[tid, bid]
```

Here's why this is challenging. When the threads in a block access `input_tensor[tid, bid]`, each one reads from a different row of the same column. In row-major layout, those memory locations are `cols` positions apart. This strided access pattern is less efficient than reading consecutive addresses.

However, this is where shared memory reduction shines: once the data is in shared memory, the reduction algorithm is identical no matter how it was loaded.

### The Reduction Tree Algorithm

```mojo
# PHASE 2: Parallel tree reduction
var stride = threads_per_block // 2
while stride > 0:
    if tid < stride:
        # Each active thread adds its partner's value
        shared[tid] += shared[tid + stride]

    # Wait for all active threads to complete this round
    barrier()

    # Move to next level of the tree
    stride //= 2
```

This is where the magic happens. The algorithm works like an upside-down binary tree (shown here for 8 threads):

- **Round 1:** Thread 0 adds thread 4's value, thread 1 adds thread 5's value, and so on.
- **Round 2:** Thread 0 adds thread 2's value, thread 1 adds thread 3's value.
- **Round 3:** Thread 0 adds thread 1's value.
- **Result:** Thread 0 holds the final sum.

The `barrier()` calls are crucial. They make sure all threads finish each round before anyone moves on to the next level.

Halving the stride only works when the number of threads is a power of two. With 12 threads, for example, some values would never be added. The updated kernels check this at compile time:

```mojo
comptime assert (
    threads_per_block & (threads_per_block - 1) == 0
), "threads_per_block must be a power of two"
```

Change `cols` to 12 in the axis=1 program and the build fails with `constraint failed: threads_per_block must be a power of two`, instead of quietly producing wrong sums.

### Block and Thread Organization

```mojo
ctx.enqueue_function[axis0_reduction_kernel](
    input_tensor,
    output_tensor,
    grid_dim=cols,  # One block per column
    block_dim=threads_per_block,  # One thread per row
)
```

For the axis=0 reduction we launch 16 blocks (one per column), each with 8 threads (one per row). Each block computes the sum of its assigned column independently.

## Impl. 2: Axis=1 Sum (Row-wise Reduction)

The second implementation handles the more straightforward case: summing across columns to produce row sums.

### The Advantage: Coalesced Memory Access

```mojo
# PHASE 1: Load data into shared memory
# Each thread loads one element from its assigned column in this row
# Neighbouring threads read neighbouring addresses: coalesced access
shared[tid] = input_tensor[bid, tid]
```

This access pattern is ideal for GPUs. When the threads in a block access `input_tensor[bid, tid]`, they read consecutive memory locations within the same row. This **coalesced access pattern** makes the best use of memory bandwidth.

(The 2025 version used `input_tensor.load[1](bid, tid)` followed by `[0]` to unwrap the value. With `TileTensor`, plain indexing does the job.)

### Identical Reduction Logic

Once the data is in shared memory, the reduction algorithm is exactly the same as in the axis=0 case:

```mojo
# PHASE 2: Parallel tree reduction
var stride = threads_per_block // 2
while stride > 0:
    if tid < stride:
        shared[tid] += shared[tid + stride]
    barrier()
    stride //= 2
```

This shows a key principle: **the reduction algorithm is independent of the data loading pattern**. Whether the data was loaded with strided or coalesced reads, the tree reduction in shared memory proceeds the same way.

### Block and Thread Organization

```mojo
ctx.enqueue_function[axis1_reduction_kernel](
    input_tensor,
    output_tensor,
    grid_dim=rows,  # One block per row
    block_dim=threads_per_block,  # One thread per column
)
```

For the axis=1 reduction we launch 8 blocks (one per row), each with 16 threads (one per column). Each block computes the sum of its assigned row independently.

## Key Design Principles and Best Practices

### 1. Shared Memory Allocation Strategy

```mojo
var shared = stack_allocation[dtype, address_space=AddressSpace.SHARED](
    row_major[threads_per_block]()
)
```

The shared memory is allocated statically, with one slot per thread in the block. Each thread writes only its own slot during the initial load, so there are no race conditions in that phase.

In Mojo 1.2, `stack_allocation` from the `layout` package returns a `TileTensor`. That means `shared[tid]` works just like indexing any other tensor, with no raw pointer arithmetic.

### 2. Synchronization Points

The `barrier()` calls are synchronization points. They ensure that:

- All threads finish loading data before the reduction begins.
- All threads in a reduction round finish before the next round starts.
- Memory stays consistent across the threads in the block.

### 3. Thread Divergence Management

```mojo
if tid < stride:
    shared[tid] += shared[tid + stride]
```

As the reduction progresses, fewer threads stay active, which causes thread divergence within a warp. That's acceptable: the active threads halve each round, so the reduction finishes in only log₂(n) rounds, and the cost is small compared to the algorithmic gain.

### 4. Memory Access Pattern Optimization

The axis=1 implementation benefits from coalesced memory access, while axis=0 has to deal with strided access. Both handle it the same way: load the data into shared memory first, then do the reduction in fast shared memory.

## Performance Considerations

**Time complexity:** The reduction phase of both implementations takes O(log n) steps, compared to O(n) for a sequential reduction.

**Memory bandwidth:** The axis=1 reduction typically performs better because of coalesced global memory access. The axis=0 reduction may use bandwidth less efficiently because of its strided access.

**Scalability:** Both implementations scale well, because each block works independently and the depth of the reduction tree grows only logarithmically.

## Complete Code

Now let's look at the complete programs that put these concepts into practice. Both have extra comments explaining the key technical decisions.

They target whatever GPU Mojo finds. That includes NVIDIA and AMD GPUs, and also Apple Silicon GPUs, which is how I tested these updates on a MacBook. If no GPU is available, the program prints a message instead of failing.

### Impl. 1: Axis=0 Sum (Column-wise Reduction)

This implementation shows how to handle the harder case of strided memory access while keeping the reduction efficient.

```mojo
from std.sys import has_accelerator
from std.testing import assert_equal
from max.gpu import thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor, ImmTileTensor, MutTileTensor, row_major, stack_allocation

comptime dtype = DType.float32
comptime rows = 8
comptime cols = 16
comptime threads_per_block = rows  # One thread per row
comptime total_elements = rows * cols

comptime input_layout = row_major[rows, cols]()
comptime output_layout = row_major[cols]()


def axis0_reduction_kernel(
    input_tensor: ImmTileTensor[dtype, type_of(input_layout), ImmUntrackedOrigin],
    output_tensor: MutTileTensor[dtype, type_of(output_layout), MutUntrackedOrigin],
):
    """Each thread block reduces one column."""
    # The halving tree below only works for power-of-two block sizes
    comptime assert (
        threads_per_block & (threads_per_block - 1) == 0
    ), "threads_per_block must be a power of two"

    # Allocate shared memory for this block's reduction
    var shared = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[threads_per_block]()
    )

    var tid = thread_idx.x  # Which row this thread handles
    var bid = block_idx.x  # Which column this block handles

    # PHASE 1: Load data into shared memory
    # Each thread loads one element from its assigned row in this column
    # NOTE: This creates a strided memory access pattern across threads
    shared[tid] = input_tensor[tid, bid]

    # Synchronize before starting reduction
    barrier()

    # PHASE 2: Parallel tree reduction
    var stride = threads_per_block // 2
    while stride > 0:
        if tid < stride:
            # Each active thread adds its partner's value
            shared[tid] += shared[tid + stride]

        # Wait for all active threads to complete this round
        barrier()

        # Move to next level of the tree
        stride //= 2

    # PHASE 3: Write result
    # Thread 0 holds the final sum for this column
    if tid == 0:
        output_tensor[bid] = shared[0]


def column_wise_sum_shared_memory(
    ctx: DeviceContext,
    input_tensor: ImmTileTensor[dtype, type_of(input_layout), _],
    output_tensor: MutTileTensor[dtype, type_of(output_layout), _],
) raises:
    """Axis=0 Sum: sum along rows (reduce rows, keep columns)."""
    print("\n" + "-" * 30)
    print("IMPLEMENTING AXIS=0 SUM")
    print("-" * 30)
    print("Strategy: Each block processes one column")
    print("Threads access elements from different rows, same column")
    print("Memory access is strided, but algorithm is still efficient")

    print(t"Launching axis=0 kernel: {cols} blocks x {threads_per_block} threads")
    ctx.enqueue_function[axis0_reduction_kernel](
        input_tensor,
        output_tensor,
        grid_dim=cols,  # One block per column
        block_dim=threads_per_block,  # One thread per row
    )


def axis_sum() raises:
    print("AXIS SUM USING PURE SHARED MEMORY REDUCTION")
    print("=" * 50)
    print(t"Working with {rows}x{cols} tensor")
    print("Demonstrate axis=0 sum operations")

    var ctx = DeviceContext()

    # Create input and output buffers on the device
    var input_buffer = ctx.enqueue_create_buffer[dtype](total_elements)
    var output_buffer = ctx.enqueue_create_buffer[dtype](cols)

    # Initialize with a clear pattern that makes verification easy
    with input_buffer.map_to_host() as host_data:
        for row in range(rows):
            for col in range(cols):
                host_data[row * cols + col] = Float32(row * 10 + col)

    # Zero the output buffer
    output_buffer.enqueue_fill(0)

    # Create tensor views over the device buffers
    var input_tensor = TileTensor(input_buffer, input_layout)
    var output_tensor = TileTensor(output_buffer, output_layout)

    # Compute expected values on the host
    var expected = List[Float32](length=cols, fill=0)
    for col in range(cols):
        for row in range(rows):
            expected[col] += Float32(row * 10 + col)

    # Axis=0 sum (sum along rows, keep columns)
    column_wise_sum_shared_memory(ctx, input_tensor.as_imm(), output_tensor)

    # Verify and display results
    with output_buffer.map_to_host() as out_host:
        print("\nout:", out_host)
        print("expected:", expected)
        for i in range(cols):
            assert_equal(out_host[i], expected[i])
        print("✓ All assertions passed!")


def main() raises:
    comptime if not has_accelerator():
        print("No GPU found")
    else:
        axis_sum()
```

Output:

```
AXIS SUM USING PURE SHARED MEMORY REDUCTION
==================================================
Working with 8x16 tensor
Demonstrate axis=0 sum operations

------------------------------
IMPLEMENTING AXIS=0 SUM
------------------------------
Strategy: Each block processes one column
Threads access elements from different rows, same column
Memory access is strided, but algorithm is still efficient
Launching axis=0 kernel: 16 blocks x 8 threads

out: HostBuffer([280.0, 288.0, 296.0, 304.0, 312.0, 320.0, 328.0, 336.0, 344.0, 352.0, 360.0, 368.0, 376.0, 384.0, 392.0, 400.0])
expected: [280.0, 288.0, 296.0, 304.0, 312.0, 320.0, 328.0, 336.0, 344.0, 352.0, 360.0, 368.0, 376.0, 384.0, 392.0, 400.0]
✓ All assertions passed!
```

### Impl. 2: Axis=1 Sum (Row-wise Reduction)

This implementation shows the ideal case, where the memory access pattern lines up perfectly with how the GPU reads memory.

```mojo
from std.sys import has_accelerator
from std.testing import assert_equal
from max.gpu import thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor, ImmTileTensor, MutTileTensor, row_major, stack_allocation

comptime dtype = DType.float32
comptime rows = 8
comptime cols = 16
comptime threads_per_block = cols  # One thread per column in the row
comptime total_elements = rows * cols

comptime input_layout = row_major[rows, cols]()
comptime output_layout = row_major[rows]()


def axis1_reduction_kernel(
    input_tensor: ImmTileTensor[dtype, type_of(input_layout), ImmUntrackedOrigin],
    output_tensor: MutTileTensor[dtype, type_of(output_layout), MutUntrackedOrigin],
):
    """Each thread block reduces one row."""
    # The halving tree below only works for power-of-two block sizes
    comptime assert (
        threads_per_block & (threads_per_block - 1) == 0
    ), "threads_per_block must be a power of two"

    # Allocate shared memory for this block's reduction
    var shared = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[threads_per_block]()
    )

    var tid = thread_idx.x  # Which column this thread handles
    var bid = block_idx.x  # Which row this block handles

    # PHASE 1: Load data into shared memory
    # Each thread loads one element from its assigned column in this row
    # Neighbouring threads read neighbouring addresses: coalesced access
    shared[tid] = input_tensor[bid, tid]

    # Synchronize to ensure all data is loaded before reduction begins
    barrier()

    # PHASE 2: Parallel tree reduction
    var stride = threads_per_block // 2
    while stride > 0:
        if tid < stride:
            # Each active thread adds its partner's value
            shared[tid] += shared[tid + stride]

        # Wait for all active threads to complete this round
        barrier()

        # Move to next level of the tree
        stride //= 2

    # PHASE 3: Write result
    # Thread 0 holds the final sum for this row
    if tid == 0:
        output_tensor[bid] = shared[0]


def row_wise_sum_shared_memory(
    ctx: DeviceContext,
    input_tensor: ImmTileTensor[dtype, type_of(input_layout), _],
    output_tensor: MutTileTensor[dtype, type_of(output_layout), _],
) raises:
    """Axis=1 Sum: sum along columns (reduce columns, keep rows)."""
    print("\n" + "-" * 30)
    print("IMPLEMENTING AXIS=1 SUM")
    print("-" * 30)
    print("Strategy: Each block processes one row independently")

    print(t"Launching axis=1 kernel: {rows} blocks x {threads_per_block} threads")
    ctx.enqueue_function[axis1_reduction_kernel](
        input_tensor,
        output_tensor,
        grid_dim=rows,  # One block per row
        block_dim=threads_per_block,  # One thread per column
    )


def axis_sum() raises:
    print("AXIS SUM USING PURE SHARED MEMORY REDUCTION")
    print("=" * 50)
    print(t"Working with {rows}x{cols} tensor")
    print("Demonstrate axis=1 sum operations")

    var ctx = DeviceContext()

    # Create input and output buffers on the device
    var input_buffer = ctx.enqueue_create_buffer[dtype](total_elements)
    var output_buffer = ctx.enqueue_create_buffer[dtype](rows)

    # Initialize with a clear pattern that makes verification easy
    with input_buffer.map_to_host() as host_data:
        for row in range(rows):
            for col in range(cols):
                host_data[row * cols + col] = Float32(row * 10 + col)

    # Zero the output buffer
    output_buffer.enqueue_fill(0)

    # Create tensor views over the device buffers
    var input_tensor = TileTensor(input_buffer, input_layout)
    var output_tensor = TileTensor(output_buffer, output_layout)

    # Compute expected values on the host
    var expected = List[Float32](length=rows, fill=0)
    for row in range(rows):
        for col in range(cols):
            expected[row] += Float32(row * 10 + col)

    # Axis=1 sum (sum along columns, keep rows)
    row_wise_sum_shared_memory(ctx, input_tensor.as_imm(), output_tensor)

    # Verify and display results
    with output_buffer.map_to_host() as out_host:
        print("\nout:", out_host)
        print("expected:", expected)
        for i in range(rows):
            assert_equal(out_host[i], expected[i])
        print("✓ All assertions passed!")


def main() raises:
    comptime if not has_accelerator():
        print("No GPU found")
    else:
        axis_sum()
```

Output:

```
AXIS SUM USING PURE SHARED MEMORY REDUCTION
==================================================
Working with 8x16 tensor
Demonstrate axis=1 sum operations

------------------------------
IMPLEMENTING AXIS=1 SUM
------------------------------
Strategy: Each block processes one row independently
Launching axis=1 kernel: 8 blocks x 16 threads

out: HostBuffer([120.0, 280.0, 440.0, 600.0, 760.0, 920.0, 1080.0, 1240.0])
expected: [120.0, 280.0, 440.0, 600.0, 760.0, 920.0, 1080.0, 1240.0]
✓ All assertions passed!
```

A few things worth pointing out in the code:

- **Kernels are plain `def` functions.** Kernel tensor arguments use *untracked* origins (`ImmUntrackedOrigin` / `MutUntrackedOrigin`), because their memory lives on the device, outside what Mojo's borrow checker can see. The input is marked `Imm` (read-only) and the output `Mut`.
- **The host-side helper uses `_` for the origin.** It accepts tensors built from a host-tracked `DeviceBuffer`, and `enqueue_function` converts them when it launches the kernel.
- **`input_tensor.as_imm()`** turns the mutable view of the input buffer into the read-only view the kernel expects.
- **The expected values are a plain `List[Float32]`.** The original used a host buffer from `enqueue_create_host_buffer`. Nothing here needs device-visible memory, so a list is simpler.

## Conclusion

These implementations show fundamental GPU programming concepts that reach far beyond axis summation. The shared memory parallel reduction pattern appears in many algorithms, including:

- Matrix reductions (finding the max, min, or mean)
- Dot products and building blocks of matrix multiplication
- Histogram computation
- Prefix sum operations

The key insight is that by loading data into shared memory and applying a tree-based reduction, we can compute efficiently in parallel regardless of the initial memory access pattern. The algorithm's elegance lies in its separation of concerns: data loading adapts to the axis being reduced, while the reduction logic stays the same.

Understanding these patterns will serve you well as you tackle more complex GPU computing challenges, from machine learning workloads to scientific computing applications.

---

## Appendix: What Changed from the 2025 Version

If you wrote code based on the original post, here's the migration checklist for Mojo 1.2.0:

| Before (2025) | Mojo 1.2.0 |
|---|---|
| `fn` | `def` (`fn` was removed) |
| `alias X = ...` | `comptime X = ...` |
| `from gpu import thread_idx, block_idx, barrier` | `from max.gpu import thread_idx, block_idx` and `from max.gpu.sync import barrier` |
| `from gpu.host import DeviceContext` | `from max.gpu.host import DeviceContext` |
| `from gpu.memory import AddressSpace` | `from max.gpu.memory import AddressSpace` |
| `from testing import ...` | `from std.testing import ...` (stdlib imports take the `std.` prefix) |
| `Layout.row_major(rows, cols)` | `row_major[rows, cols]()` (a free function) |
| `LayoutTensor[dtype, layout, MutableAnyOrigin]` | `TileTensor`: `ImmTileTensor[dtype, type_of(layout), ImmUntrackedOrigin]` for kernel inputs, `MutTileTensor[..., MutUntrackedOrigin]` for outputs |
| `InputTensor(input_buffer)` | `TileTensor(input_buffer, layout)` |
| `stack_allocation[N, Scalar[dtype], address_space=...]()` from `memory` | `stack_allocation[dtype, address_space=...](row_major[N]())` from `layout`, which returns a `TileTensor` |
| `input_tensor[tid, bid][0]` | `input_tensor[tid, bid]` |
| `input_tensor.load[1](bid, tid)` then `[0]` | `input_tensor[bid, tid]` |
| `String("{}x{}").format(rows, cols)` | `t"{rows}x{cols}"` (t-strings) |
| `_ = output_buffer.enqueue_fill(0)` | `output_buffer.enqueue_fill(0)` |
| Trailing `;` on every line | No semicolons (they were legal but not idiomatic) |
| No GPU check | `comptime if not has_accelerator(): ...` |

The GPU packages (`max.gpu`, `layout`) ship with MAX, so install `max` alongside `mojo` (for example `pixi add max` or `pip install modular`). Then run each program with `mojo run axis0_sum.mojo` / `mojo run axis1_sum.mojo`.
