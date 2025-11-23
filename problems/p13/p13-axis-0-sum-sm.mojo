from gpu import thread_idx, block_idx, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from math import iota
from gpu.memory import AddressSpace
from memory import stack_allocation
from testing import assert_equal

alias dtype = DType.float32;
alias rows = 8;
alias cols = 16;
alias threads_per_block = rows;
alias total_elements = rows * cols;

fn axis_sum() raises:
    """
    Complete demonstration of axis sum using only shared memory parallel reduction.
    We'll implement both axis=0 and axis=1 to show how the algorithm adapts.
    """
    print("AXIS SUM USING PURE SHARED MEMORY REDUCTION");
    print("="*50);
    print(String("Working with {}x{} tensor").format(rows, cols));
    print("Demonstrate axis=0 sum operations");

    var ctx = DeviceContext();

    # Create input tensor buffer
    var input_buffer = ctx.enqueue_create_buffer[dtype](total_elements);

    # Create output buffers for both axis operations
    var output_buffer = ctx.enqueue_create_buffer[dtype](cols);

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

    alias output_layout = Layout.row_major(cols);
    alias OutputTensor = LayoutTensor[dtype, output_layout, MutableAnyOrigin];
    var output_tensor = OutputTensor(output_buffer);

    var expected_buffer = ctx.enqueue_create_host_buffer[dtype](cols);
    ctx.synchronize();

    # Fill expected buffer with calculated values
    for i in range(cols):
        var expected: Float32 = 0.0;
        for j in range(rows):
            expected += Float32(j * 10 + i);
        expected_buffer[i] = expected;
    
    # Let's implement axis=0 sum (sum along rows, keep columns)
    column_wise_sum_shared_memory(
        ctx,
        input_tensor,
        output_tensor,
    );

    # Verify and display results
    with output_buffer.map_to_host() as output_buffer_host:
        print("\nout:", output_buffer_host);
        print("expected:", expected_buffer);

        for i in range(cols):
            assert_equal(output_buffer_host[i], expected_buffer[i]);
        
        print("✓ All assertions passed!");


fn column_wise_sum_shared_memory(
    ctx: DeviceContext,
    input_tensor: LayoutTensor[dtype, Layout.row_major(rows, cols), MutableAnyOrigin],
    output_tensor: LayoutTensor[dtype, Layout.row_major(cols), MutableAnyOrigin]
) raises:
    """
    Axis=0 Sum: Sum along rows (reduce rows, keep columns).

    This is more complex because we need to sum elements from different rows
    but the same column. The memory access pattern is less optimal, but
    shared memory reduction still works beautifully.    
    """
    print("\n" + "-"*30);
    print("IMPLEMENTING AXIS=0 SUM");
    print("-"*30);
    print("Strategy: Each block processes one column");
    print("Threads access elements from different rows, same column");
    print("Memory access is strided, but algorithm is still efficient");

    fn axis0_reduction_kernel(
        input_tensor : LayoutTensor[dtype, Layout.row_major(rows, cols), MutableAnyOrigin],
        output_tensor : LayoutTensor[dtype, Layout.row_major(cols), MutableAnyOrigin] 
    ):
        """
        Each thread block reduces one row.
        This is a direct application of our shared memory reduction pattern.
        """
        # Allocate shared memory for this blocks's reduction
        var shared = stack_allocation[
            threads_per_block,
            Scalar[dtype],
            address_space=AddressSpace.SHARED,
        ]();

        var tid = thread_idx.x;
        var bid = block_idx.x;

        # PHASE 1: Load data into shared memory
        # Each thread loads one element from its assigned row in this column
        # NOTE: This creates a strided memory access pattern across threads
        shared[tid] = input_tensor[tid, bid][0];

        # Synchronize before starting reduction
        barrier();

        # PHASE 2: Parallel reduction - same algorithm as axis=1!
        # The beauty of shared memory reduction: once data is in shared memory,
        # the reduction algorithm is identical regardless of how we loaded the data
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
    
    print("Launching axis=0 kernel: 16 blocks x 8 threads");
    ctx.enqueue_function[axis0_reduction_kernel](
        input_tensor,
        output_tensor,
        grid_dim=cols,                  # One block per column
        block_dim=threads_per_block,    # One thread per row
    );



fn main() raises:
    axis_sum();