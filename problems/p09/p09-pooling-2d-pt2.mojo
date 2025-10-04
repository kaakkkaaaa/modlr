from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from builtin.math import max
from utils.numerics import neg_inf

# Configuration for LayoutTensor approach
alias TPB_X = 8;
alias TPB_Y = 8;
alias INPUT_H = 8;
alias INPUT_W = 8;
alias POOL_SIZE = 2;
alias STRIDE = 2;
alias OUTPUT_H = INPUT_H // STRIDE; # 4
alias OUTPUT_W = INPUT_W // STRIDE; # 4

alias BLOCKS_PER_GRID = (1, 1);
alias THREADS_PER_BLOCK = (TPB_X, TPB_Y);
alias dtype = DType.float32;

alias input_layout = Layout.row_major(INPUT_H, INPUT_W);
alias output_layout = Layout.row_major(OUTPUT_H, OUTPUT_W);

fn max_pool_2d[
    input_layout: Layout,
    output_layout: Layout,
](
    input: LayoutTensor[dtype, input_layout, MutableAnyOrigin],
    output: LayoutTensor[dtype, output_layout, MutableAnyOrigin],
    pool_h: Int,
    pool_w: Int,
    stride_h: Int,
    stride_w: Int,
):
    # 2D Max Pooling using LayoutTensor
    var input_h = input.dim(0); # height
    var input_w = input.dim(1); # width
    var output_h = output.dim(0);
    var output_w = output.dim(1);

    # Each thread computes one output pixel
    var out_y = block_dim.y * block_idx.y + thread_idx.y;
    var out_x = block_dim.x * block_idx.x + thread_idx.x;

    # Check if this thread doesn't have work to do
    if out_y >= output_h or out_x >= output_w:
        return
    
    # Find the starting position in the input for this output pixel
    var start_y = out_y * stride_h;
    var start_x = out_x * stride_w;

    # Initialize with negative infinity
    var max_val: SIMD[dtype, 1] = SIMD[dtype, 1](-3.4028235e+38);

    # Scan through the pooling window
    for pool_y in range(pool_h):
        for pool_x in range(pool_w):
            var input_y = start_y + pool_y;
            var input_x = start_x + pool_x;

            # Bounds checking
            if input_y < input_h and input_x < input_w:
                # Here's the magic! Clean 2D indexing with LayoutTensor
                var current_val = input.load[1](input_y, input_x);
                if current_val > max_val:
                    max_val = current_val;
    
    # Store the result
    output[out_y, out_x] = max_val;


fn main() raises:
    with DeviceContext() as ctx:
        inp_buf = ctx.enqueue_create_buffer[dtype](INPUT_H * INPUT_W).enqueue_fill(0);
        out_buf = ctx.enqueue_create_buffer[dtype](OUTPUT_H * OUTPUT_W).enqueue_fill(0);

        # Initialize with a simple pattern
        with inp_buf.map_to_host() as inp_buf_host:
            for i in range(INPUT_H * INPUT_W):
                var y = i // INPUT_W;
                var x = i % INPUT_W;
                inp_buf_host[i] = y * 10 + x;
        
        inp_tensor = LayoutTensor[dtype, input_layout, MutableAnyOrigin](inp_buf.unsafe_ptr());
        out_tensor = LayoutTensor[dtype, output_layout, MutableAnyOrigin](out_buf.unsafe_ptr());

        # Launch kernels
        ctx.enqueue_function[max_pool_2d[input_layout, output_layout]](
            inp_tensor,
            out_tensor,
            POOL_SIZE, POOL_SIZE,
            STRIDE, STRIDE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        );

        ctx.synchronize();

        # Print results to show it works
        with inp_buf.map_to_host() as inp_buf_host:
            print("Input (8x8):");
            for y in range(INPUT_H):
                for x in range(INPUT_W):
                    print(String("{}").format(inp_buf_host[y * INPUT_W + x]), end=" ");
                print();
        
        with out_buf.map_to_host() as out_buf_host:
            print("\nOutput (4x4) after max pooling:");
            for y in range(OUTPUT_H):
                for x in range(OUTPUT_W):
                    print(String("{}").format(out_buf_host[y * OUTPUT_W + x]), end=" ");
                print();