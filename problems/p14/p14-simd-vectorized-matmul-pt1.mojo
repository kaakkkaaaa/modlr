from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_equal
from time import perf_counter_ns

alias TPB = 32;
alias SIZE = 32;
alias BLOCKS_PER_GRID = (1, 1);
alias THREADS_PER_BLOCK = (TPB, TPB);
alias TILE_SIZE = 16;
alias dtype = DType.float32;
alias layout = Layout.row_major(SIZE, SIZE);

fn simd_vectorized_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    """
    SIMD vectorized matrix multiplication.
    """

    alias VEC_WIDTH = 4;

    var tx = thread_idx.x;
    var ty = thread_idx.y;
    var row = block_idx.y * TILE_SIZE + ty;
    var base_col = block_idx.x * TILE_SIZE + tx * VEC_WIDTH;

    # SIMD accumulator
    var vec_sum = SIMD[dtype, VEC_WIDTH](0.0);

    if row < size and base_col < size:
        for k in range(size):
            var a_val = a.load[1](row, k);
            # Create SIMD vector from B matrix elements
            var remaining = min(VEC_WIDTH, size - base_col);
            var b_vec = SIMD[dtype, VEC_WIDTH](0.0);

            for i in range(remaining):
                if base_col + i < size:
                    b_vec[i] = b.load[1](k, base_col + i);
            
            # SIMD multiply-accumulate
            vec_sum += a_val * b_vec;
        
        # Store results back to global memory
        var remaining = min(VEC_WIDTH, size - base_col);
        for i in range(remaining):
            if base_col + i < size:
                output.store[1](row, base_col + i, vec_sum[i]);

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
            
        out_tensor = LayoutTensor[mut=False, dtype, layout](out.unsafe_ptr());
        a_tensor = LayoutTensor[mut=False, dtype, layout](inp1.unsafe_ptr());
        b_tensor = LayoutTensor[mut=False, dtype, layout](inp2.unsafe_ptr());

        ctx.enqueue_function[simd_vectorized_matmul[layout, SIZE]](
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