from gpu import thread_idx, block_idx, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from math import iota
from gpu.memory import AddressSpace
from memory import stack_allocation
from testing import assert_equal, assert_almost_equal

alias dtype = DType.float32;
alias rows = 8;
alias cols = 16;
alias total_elements = rows * cols;

fn axis_sum() raises:
    """
    Complete demonstration of axis sum using only shared memory parallel reduction.
    We'll implement both axis=0 and axis=1 to show how the algorithm adapts.
    """
    print("AXIS SUM USING PURE SHARED MEMORY REDUCTION");
    print("="*50);
    print(String("Working with {}x{} tensor").format(rows, cols));
    print("Will demonstrate axis=1 sum operations");

    var ctx = DeviceContext();

    # Create input tensor buffer
    var input_buffer = ctx.enqueue_create_buffer[dtype](total_elements);

    # Create output buffers for both axis operations
    var output_buffer = ctx.enqueue_create_buffer[dtype](rows);

    # Initialize with a clear pattern that makes verification easy
    with input_buffer.map_to_host() as host_data:
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col;
                # Create a pattern where each row has predictable sums
                host_data[idx] = Float32(row * 10 + col);

    # Zero the output buffers
    _ = output_buffer.enqueue_fill(0);

    # Create tensor views
    alias tensor_layout = Layout.row_major(rows, cols);
    alias InputTensor = LayoutTensor[dtype, tensor_layout, MutableAnyOrigin];
    var input_tensor = InputTensor(input_buffer);

    alias output_layout = Layout.row_major(rows);
    alias OutputTensor = LayoutTensor[dtype, output_layout, MutableAnyOrigin];
    var output_tensor = OutputTensor(output_buffer);

    var expected_buffer = ctx.enqueue_create_host_buffer[dtype](rows);
    ctx.synchronize();

    # Fill expected buffer with calculated values
    for i in range(rows):
        var expected: Float32 = 0.0;
        for j in range(cols):
            expected += Float32(i * 10 + j);
        expected_buffer[i] = expected;

    # Let's implement axis=1 sum (sum along columns, keep rows)
    row_wise_sum_shared_memory(
        ctx,
        input_tensor,
        output_tensor
    );

    # Verify and display results
    with output_buffer.map_to_host() as output_buffer_host:
        print("\nout:", output_buffer_host);
        print("expected:", expected_buffer);

        for i in range(rows):
            assert_equal(output_buffer_host[i], expected_buffer[i]);
        
        print("✓ All assertions passed!");

fn row_wise_sum_shared_memory(
    ctx: DeviceContext, 
    input_tensor : LayoutTensor[dtype, Layout.row_major(rows, cols), MutableAnyOrigin], 
    output_tensor : LayoutTensor[dtype, Layout.row_major(rows), MutableAnyOrigin]
) raises:
    """
    Axis=1 Sum: Sum along columns (reduce columns, keep rows)
    This is the "easy" case because each row can be processed independently.
    Memory access pattern: threads access consecutive memory locations.
    """

    print("\n" + "-"*30);
    print("IMPLEMENTING AXIS=1 SUM");
    print("-"*30);
    print("Strategy: Each block processes one row independently");

    alias threads_per_block = cols; # One thread per column in the row

    fn axis1_reduction_kernel(
        input_tensor: LayoutTensor[dtype, Layout.row_major(rows, cols), MutableAnyOrigin], 
        output_tensor: LayoutTensor[dtype, Layout.row_major(rows), MutableAnyOrigin]
    ):
        """
        Each thread block reduces one row.
        This is a direct application of our shared memory reduction pattern.
        """
        # Allocate shared memory for this block's reduction
        var shared = stack_allocation[
            threads_per_block,
            Scalar[dtype],
            address_space=AddressSpace.SHARED,
        ]();

        var tid = thread_idx.x;     # Which column this thread handles
        var bid = block_idx.x;      # Which row this block handles

        # PHASE 1: Load data into shared memory
        # Each thread loads one element from its assigned column in this row
        var loaded_value = input_tensor.load[1](bid, tid);
        shared[tid] = loaded_value[0];
        
        # Synchronize to ensure all data is loaded before reduction begins
        barrier();

        # PHASE 2: Parallel reduction using the tree pattern
        # This is exactly our standard shared memory reduction algorithm
        var stride = threads_per_block // 2;
        while stride > 0:
            if tid < stride:
                # Each active thread adds its partner's value
                shared[tid] += shared[tid + stride];
            
            # Wait for all active threads to complete this round
            barrier();

            # Move to next level of the tree
            stride //= 2;
        
        # PHASE 3: Write result
        # Thread 0 holds the final sum for this row
        if tid == 0:
            output_tensor[bid] = shared[0];
        
    print("Launching axis=1 kernel: 8 blocks x 16 threads");
    ctx.enqueue_function[axis1_reduction_kernel](
        input_tensor,
        output_tensor,
        grid_dim=rows,  # One block per row
        block_dim=threads_per_block,    # One thread per column
    );

fn main() raises:
    axis_sum();