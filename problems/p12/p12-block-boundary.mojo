from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from math import log2, ceildiv
from testing import assert_equal

# ANCHOR: prefix_sum_complete
alias TPB = 8;
alias SIZE = 15;
alias BLOCKS_PER_GRID = (ceildiv(SIZE , TPB), 1);
alias THREADS_PER_BLOCK = (TPB, 1);
alias dtype = DType.float32;
# alias EXTENDED_SIZE = SIZE + 2;
alias layout = Layout.row_major(SIZE);

# Kernel 1: Local prefix sum within each block
fn prefix_sum_local_phase[
    out_layout: Layout, in_layout: Layout
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    size: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x;
    var local_i = thread_idx.x;

    # Use TPB for shared memory size (not EXTENDED_SIZE)
    shared_mem = tb[dtype]().row_major[TPB]().shared().alloc();
    if global_i < size:
        shared_mem[local_i] = a[global_i];
    else:
        shared_mem[local_i] = 0.0;
    
    barrier();

    offset = 1;
    for i in range(Int(log2(Scalar[dtype](TPB)))):
        var current_val: output.element_type = 0.0;
        if local_i >= offset and local_i < size:
            current_val = shared_mem[local_i - offset];
        barrier();
        if local_i >= offset and local_i < size:
            shared_mem[local_i] += current_val;
        barrier();
        offset *= 2;
    
    if global_i < size:
        output[global_i] = shared_mem[local_i];

# Kernel 2: Add block sums to each element
fn prefix_sum_block_sum_phase[
    layout: Layout
](
    output: LayoutTensor[mut=True, dtype, layout],
    size: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x;

    # Add the total sum of the previous block to each element in the current block
    if block_idx.x > 0 and global_i < size:
        var prev_block_sum: output.element_type = output[block_idx.x * TPB - 1];
        output[global_i] += prev_block_sum;
    
# ANCHOR_END: prefix_sum_complete
fn main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0.0);
        var a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0.0);

        # Initialize input data
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = i;
        
        var a_tensor = LayoutTensor[mut=False, dtype, layout](
            a.unsafe_ptr()
        );

        var out_tensor = LayoutTensor[mut=True, dtype, layout](
            out.unsafe_ptr()
        );

        # Phase 1: Local prefix sum within each block
        ctx.enqueue_function[prefix_sum_local_phase[layout, layout]](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        );

        ctx.synchronize();

        # Phase 2: Add block sums
        ctx.enqueue_function[prefix_sum_block_sum_phase[layout]](
            out_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        );

        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(0.0);
        ctx.synchronize();

        with a.map_to_host() as a_host:
            expected[0] = a_host[0];
            for i in range(1, SIZE):
                expected[i] = expected[i - 1] + a_host[i];
        
        # Verify results
        with out.map_to_host() as out_host:
            print("Output:", out_host);
            print("Expected:", expected);

            # Check each element
            var success = True;
            for i in range(SIZE):
                if out_host[i] != expected[i]:
                    success = False;
                    print("Mismatch at index", i, ": got", out_host[i], "expected", expected[i]);
            if success:
                print("Test passed!");
            else:
                print("Test failed!");

                
