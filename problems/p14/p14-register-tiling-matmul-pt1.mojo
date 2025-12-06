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

alias REG_TILE_SIZE = 4;
alias TILE_SIZE = 16;

fn register_tiling_matmul[
    layout: Layout,
    size: Int
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    """
    Register tiling using only native GPU programming in Mojo. 
    Each thread computes REG_TILE_SIZE consecutive elements.
    """

    var tx = thread_idx.x;
    var ty = thread_idx.y;

    # Calculate which elements this thread computes
    var base_row = block_idx.y * TILE_SIZE + ty * REG_TILE_SIZE;
    var col = block_idx.x * TILE_SIZE + tx;
    
    # Explicit register variables (guaranteed to be in registers)
    var acc0 = SIMD[dtype, 1](0.0);
    var acc1 = SIMD[dtype, 1](0.0);
    var acc2 = SIMD[dtype, 1](0.0);
    var acc3 = SIMD[dtype, 1](0.0);

    # Simple dot product computation (no shared memory for simplicity)
    if col < size:
        for k in range(size):
            var b_val = b.load[1](k, col);

            # Compute 4 dot products simultaneously
            if base_row + 0 < size:
                var a0 = a.load[1](base_row + 0, k);
                acc0 += a0 * b_val;
            if base_row + 1 < size:
                var a1 = a.load[1](base_row + 1, k);
                acc1 += a1 * b_val;
            if base_row + 2 < size:
                var a2 = a.load[1](base_row + 2, k);
                acc2 += a2 * b_val;
            if base_row + 3 < size:
                var a3 = a.load[1](base_row + 3, k);
                acc3 += a3 * b_val;
    
    # Store results back to global memory
    if base_row + 0 < size and col < size:
        output.store[1](base_row + 0, col, acc0);
    if base_row + 1 < size and col < size:
        output.store[1](base_row + 1, col, acc1);
    if base_row + 2 < size and col < size:
        output.store[1](base_row + 2, col, acc2);
    if base_row + 3 < size and col < size:
        output.store[1](base_row + 3, col, acc3);
    
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

        ctx.enqueue_function[register_tiling_matmul[layout, SIZE]](
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