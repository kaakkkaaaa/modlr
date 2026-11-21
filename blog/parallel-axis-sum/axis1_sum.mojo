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
