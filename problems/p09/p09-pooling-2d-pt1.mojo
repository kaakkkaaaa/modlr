from memory import UnsafePointer, stack_allocation
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from sys import sizeof
from testing import assert_equal

# Configuration for 2D pooling
alias TPB_X = 8;    # threads per block in X dimension
alias TPB_Y = 8;    # threads per block in Y dimension
alias INPUT_H = 8;  # input height
alias INPUT_W = 8;  # input width
alias POOL_SIZE = 2;    # 2x2 pooling window
alias STRIDE = 2;   # stride of 2 (non-overlapping windows)
alias OUTPUT_H = INPUT_H // STRIDE; # 4
alias OUTPUT_W = INPUT_W // STRIDE; # 4

alias BLOCKS_PER_GRID = (1, 1);
alias THREADS_PER_BLOCK = (TPB_X, TPB_Y);
alias dtype = DType.float32;

fn max_pool_2d(
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    input_h: Int,
    input_w: Int,
    pool_h: Int,
    pool_w: Int,
    stride_h: Int,
    stride_w: Int,
):
    """
    2D Max Pooling using raw memory approach.
    Each thread computes one output pixel by examining a pool_h x pool_w window.
    """

    # Calculate which output pixel this thread should compute
    var out_y = block_dim.y * block_idx.y + thread_idx.y;
    var out_x = block_dim.x * block_idx.x + thread_idx.x;

    # Check if this thread has valid work to do
    if out_y >= OUTPUT_H or out_x >= OUTPUT_W:
        return
    
    # Calculate the starting position in the input for this output pixel
    var start_y = out_y * stride_h;
    var start_x = out_x * stride_w;

    # Initialize with negative infinity (or very small value)
    var max_value = Scalar[dtype](1e-10);

    # Iterate through the pooling window
    for pool_y in range(pool_h):
        for pool_x in range(pool_w):
            var input_y = start_y + pool_y;
            var input_x = start_x + pool_x;

            # Bounds checking
            if input_y < input_h and input_x < input_w:
                var input_idx = input_y * input_w + input_x;
                var current_val = input[input_idx];

                # Update maximum
                if current_val > max_value:
                    max_value = current_val;
    
    # Store the result
    var output_idx = out_y * OUTPUT_W + out_x;
    output[output_idx] = max_value;


fn main() raises:
    with DeviceContext() as ctx:
        # Create input and output buffers
        var input_size = INPUT_H * INPUT_W;
        var output_size = OUTPUT_H * OUTPUT_W;

        var input_buf = ctx.enqueue_create_buffer[dtype](input_size).enqueue_fill(0);
        var max_output = ctx.enqueue_create_buffer[dtype](output_size).enqueue_fill(0);
        
        # Initialize input with some test pattern
        with input_buf.map_to_host() as input_buf_host:
            for i in range(input_size):
                input_buf_host[i] = i; # Values 0-9 repeating
        
        # Launch kernels
        ctx.enqueue_function[max_pool_2d](
            max_output.unsafe_ptr(),
            input_buf.unsafe_ptr(),
            INPUT_H, INPUT_W,
            POOL_SIZE, POOL_SIZE,
            STRIDE, STRIDE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        );

        ctx.synchronize();

        # Print results
        with input_buf.map_to_host() as input_host:
            print("Input (8x8):");
            for y in range(INPUT_H):
                for x in range(INPUT_W):
                    print(input_host[y * INPUT_W + x], end=" ");
                print();

        with max_output.map_to_host() as max_output_host:
            print("\nMax Pooling Result (4x4):");
            for y in range(OUTPUT_H):
                for x in range(OUTPUT_W):
                    print(max_output_host[y * OUTPUT_W + x], end=" ");
                print();
