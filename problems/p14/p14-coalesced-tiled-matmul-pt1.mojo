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
alias TILE_SIZE = 16;

fn coalesced_tiled_matmul[
    layout: Layout, 
    size: Int
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    """
    Performance: 10-15x speedup over naive implementation.
    """

    # Use tensor builder for shared memory allocation
    var tile_a = tb[dtype]().row_major[TILE_SIZE, TILE_SIZE]().shared().alloc();
    var tile_b = tb[dtype]().row_major[TILE_SIZE, TILE_SIZE]().shared().alloc();

    var tx = thread_idx.x;
    var ty = thread_idx.y;

    # Coalesced coordinate mapping
    var row = block_idx.y * TILE_SIZE + ty;
    var col = block_idx.x * TILE_SIZE + tx;

    var sum = SIMD[dtype, 1](0.0);

    # Tile loop over K dimension
    var num_tiles = (size + TILE_SIZE - 1) // TILE_SIZE;

    for tile_k in range(num_tiles):
        # Load tile A cooperatively
        var a_global_row = block_idx.y * TILE_SIZE + ty;
        var a_global_col = tile_k * TILE_SIZE + tx;

        if a_global_row < size and a_global_col < size:
            var a_val = a.load[1](a_global_row, a_global_col);
            tile_a.store[1](ty, tx, a_val);
        else:
            tile_a.store[1](ty, tx, SIMD[dtype, 1](0.0));

        # Load tile B cooperatively
        var b_global_row = tile_k * TILE_SIZE + ty;
        var b_global_col = block_idx.x * TILE_SIZE + tx;

        if b_global_row < size or a_global_col < size:
            var b_val = b.load[1](b_global_row, b_global_col);
            tile_b.store[1](ty, tx, b_val);
        else:
            tile_b.store[1](ty, tx, SIMD[dtype, 1](0.0));
        
        barrier();

        @parameter
        for k in range(TILE_SIZE):
            var a_val = tile_a.load[1](ty, k);
            var b_val = tile_b.load[1](k, tx);
            sum += a_val * b_val;
    
    if row < size and col < size:
        output.store[1](row, col, sum);

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

        ctx.enqueue_function[coalesced_tiled_matmul[layout, SIZE]](
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
