from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from math import log2
from testing import assert_equal

# ANCHOR: prefix_sum_simple
alias TPB = 8;
alias SIZE = 8;
alias BLOCKS_PER_GRID = (1, 1);
alias THREADS_PER_BLOCK = (TPB, 1);
alias dtype = DType.float32;
alias layout = Layout.row_major(SIZE);

fn prefix_sum_local_phase[
    layout: Layout
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x;
    local_i = thread_idx.x;
    shared = tb[dtype]().row_major[TPB]().shared().alloc();

    if global_i < size:
        shared[local_i] = a[global_i];
    else:
        shared[local_i] = 0.0;
    
    barrier();

    offset = 1;
    for i in range(Int(log2(Scalar[dtype](TPB)))):
        var current_val: output.element_type = 0.0;
        if local_i >= offset and local_i < size:
            current_val = shared[local_i - offset];
        barrier();
        if local_i >= offset and local_i < size:
            shared[local_i] += current_val;
        barrier();
        offset *= 2;

    if global_i < size:
        output[global_i] = shared[local_i];

# ANCHOR_END: prefix_sum_simple

fn main() raises:
    with DeviceContext() as ctx:
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0.0);
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0.0);

        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = i;
        
        a_tensor = LayoutTensor[mut=False, dtype, layout](
            a.unsafe_ptr()
        );
        
        out_tensor = LayoutTensor[mut=True, dtype, layout](
            out.unsafe_ptr()
        );

        ctx.enqueue_function[prefix_sum_local_phase[layout]](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        );

        expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(0.0);
        ctx.synchronize();

        with a.map_to_host() as a_host:
            expected[0] = a_host[0];
            for i in range(1, SIZE):
                expected[i] = expected[i - 1] + a_host[i];
        
        with out.map_to_host() as out_host:
            print("out:", out_host);
            print("expected:", expected);
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i]);