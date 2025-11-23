from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from testing import assert_equal

# ANCHOR: conv_1d_block_boundary
alias TPB = 8;
alias SIZE = 15;
alias CONV = 4;
alias BLOCKS_PER_GRID = (2, 1);
alias THREADS_PER_BLOCK = (TPB, 1);
alias dtype = DType.float32;
alias in_layout = Layout.row_major(SIZE);
alias out_layout = Layout.row_major(SIZE);
alias conv_layout = Layout.row_major(CONV);

fn conv_1d_block_boundary[
    in_layout: Layout, out_layout: Layout, conv_layout: Layout,
](
    output: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, conv_layout],
):
    global_i = block_idx.x * block_dim.x + thread_idx.x;
    local_i = thread_idx.x;
    # first: need to account for padding
    shared_a = tb[dtype]().row_major[TPB + CONV - 1]().shared().alloc();
    shared_b = tb[dtype]().row_major[CONV]().shared().alloc();

    if global_i < SIZE:
        shared_a[local_i] = a[global_i];
    else:
        shared_a[local_i] = 0.0;
    
    # second: load elements needed for convolution at block boundary 
    if local_i < CONV - 1:
        # indices from next block
        next_idx = global_i + TPB;
        if next_idx < SIZE:
            shared_a[TPB + local_i] = a[next_idx];
        else:
            # Initialize out-of-bounds elements to 0 to avoid reading from uninitialized memory
            # which is an undefined behavior
            shared_a[TPB + local_i] = 0.0;
    
    if local_i < CONV:
        shared_b[local_i] = b[local_i];
    
    barrier();

    if global_i < SIZE:
        var local_sum: output.element_type = 0;

        @parameter
        for j in range(CONV):
            if global_i + j < SIZE:
                local_sum += shared_a[local_i + j] * shared_b[j];
        output[global_i] = local_sum;
    
# ANCHOR_END: conv_1d_block_boundary

fn main() raises:
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0);
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0);
        b = ctx.enqueue_create_buffer[dtype](CONV).enqueue_fill(0);
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = i;
        
        with b.map_to_host() as b_host:
            for i in range(CONV):
                b_host[i] = i;
        
        out_tensor = LayoutTensor[mut=False, dtype, out_layout](
            out.unsafe_ptr()
        );

        a_tensor = LayoutTensor[mut=False, dtype, in_layout](
            a.unsafe_ptr()
        );

        b_tensor = LayoutTensor[mut=False, dtype, conv_layout](
            b.unsafe_ptr()
        );

        ctx.enqueue_function[
            conv_1d_block_boundary[in_layout, out_layout, conv_layout]
        ](
            out_tensor,
            a_tensor,
            b_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        );

        ctx.synchronize();

        expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(0);

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                for j in range(CONV):
                    if i + j < SIZE:
                        expected[i] += a_host[i + j] * b_host[j];
        
        with out.map_to_host() as out_host:
            print("out:", out_host);
            print("expected:", expected);
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i]);


        
