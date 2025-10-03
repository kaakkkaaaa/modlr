from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_equal
from time import perf_counter_ns

alias TPB = 3;
alias SIZE = 2;
alias BLOCKS_PER_GRID = (1, 1);
alias THREADS_PER_BLOCK = (TPB, TPB);
alias dtype = DType.float32;
alias layout = Layout.row_major(SIZE, SIZE);

fn single_block_matmul[
    layout: Layout,
    size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    # Calculate thread positions
    row = block_dim.y * block_idx.y + thread_idx.y;
    col = block_dim.x * block_idx.x + thread_idx.x;
    local_row = thread_idx.y;
    local_col = thread_idx.x;

    # CRITICAL: Exit early if thread is out of bounds for shared memory
    if local_row >= size or local_col >= size:
        return

    # Allocate shared memory for tiles
    var a_shared = tb[dtype]().row_major[size, size]().shared().alloc();
    var b_shared = tb[dtype]().row_major[size, size]().shared().alloc();

    # Initialize result accumulator
    var result = SIMD[dtype, 1](0.0);

    # Calculate number of tiles needed along K dimension
    var num_tiles = (a.dim(1) + size - 1) // size;

    # Process each tile along K dimension
    for tile in range(num_tiles):
        # PHASE 1: Load tile from global to shared memory
        if row < a.dim(0) and (tile * size + local_col) < a.dim(1):
            a_shared[local_row, local_col] = a[row, tile * size + local_col][0];
        else:
            a_shared[local_row, local_col] = 0.0;
        
        if (tile * size + local_row) < b.dim(0) and col < b.dim(1):
            b_shared[local_row, local_col] = b[tile * size + local_row, col][0];
        else:
            b_shared[local_row, local_col] = 0.0;
        
        # PHASE 2: Synchronize all threads
        barrier();

        # PHASE 3: Compute using shared memory data
        @parameter
        for k in range(size):
            result += a_shared[local_row, k][0] * b_shared[k, local_col][0];
        
        # PHASE 4: Synchronize before nex iteration
        barrier();
    
    # PHASE 5: Write final result to global memory
    if row < output.dim(0) and col < output.dim(1):
        output[row, col] = result;

fn main() raises:
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](
            SIZE * SIZE).enqueue_fill(0);
        inp1 = ctx.enqueue_create_buffer[dtype](
            SIZE * SIZE).enqueue_fill(0);
        inp2 = ctx.enqueue_create_buffer[dtype](
            SIZE * SIZE).enqueue_fill(0);
        
        expected = ctx.enqueue_create_host_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(0);

        with inp1.map_to_host() as inp1_host, inp2.map_to_host() as inp2_host:
            for row in range(SIZE):
                for col in range(SIZE):
                    val = row * SIZE + col;
                    # row major: placing elements row by row
                    inp1_host[row * SIZE + col] = val;
                    inp2_host[row * SIZE + col] = Float32(2.0) * val;
            
            # inp1 @ inp2.T
            for i in range(SIZE):
                for j in range(SIZE):
                    for k in range(SIZE):
                        expected[i * SIZE + j] += (
                            inp1_host[i * SIZE + k] * inp2_host[k * SIZE + j]
                        );
            
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr());
        a_tensor = LayoutTensor[mut=False, dtype, layout](inp1.unsafe_ptr());
        b_tensor = LayoutTensor[mut=False, dtype, layout](inp2.unsafe_ptr());

        ctx.enqueue_function[single_block_matmul[layout, SIZE]](
            out_tensor,
            a_tensor,
            b_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        );

        ctx.synchronize();

        var start_time = perf_counter_ns();

        with out.map_to_host() as out_host:
            print("out:", out_host);
            print("expected:", expected);
            for col in range(SIZE):
                for row in range(SIZE):
                    assert_equal(
                        out_host[col * SIZE + row], expected[col * SIZE + row]
                    );

        var end_time = perf_counter_ns();

        # Calculate and display timing results
        var elapsed_ns = end_time - start_time;
        var elapsed_ms = Float64(elapsed_ns) / 1_000_000.0;

        # print(String("Validation time: {} nanoseconds").format(elapsed_ns));
        print(String("Elapsed time: {} ms").format(elapsed_ms));
