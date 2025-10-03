from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_equal
from time import perf_counter_ns

alias TPB = 3;
alias SIZE_TILED = 9;
alias dtype = DType.float32;
alias BLOCKS_PER_GRID_TILED = (3, 3);   # Each block covers 3x3 elements
alias THREADS_PER_BLOCK_TILED = (TPB, TPB);
alias layout_tiled = Layout.row_major(SIZE_TILED, SIZE_TILED);

fn matmul_tiled[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    local_row = thread_idx.y;
    local_col = thread_idx.x;
    tiled_row = block_idx.y * TPB + thread_idx.y;
    tiled_col = block_idx.x * TPB + thread_idx.x;

    sm_tile_a = tb[dtype]().row_major[TPB, TPB]().shared().alloc();
    sm_tile_b = tb[dtype]().row_major[TPB, TPB]().shared().alloc();
    
    var sum = SIMD[dtype, 1](0.0);

    # Iterate over tiles to compute matrix product
    @parameter
    for tile in range((size + TPB - 1) // TPB):
        # Load A tile - global row stays the same, col determined by tile
        if tiled_row < size and (tile * TPB + local_col) < size:
            sm_tile_a[local_row, local_col] = a.load[1](
                tiled_row, tile * TPB + local_col
            );
        
        # Load B tile - row determined by tile, global col stays the same
        if (tile * TPB + local_row) < size and tiled_col < size:
            sm_tile_b[local_row, local_col] = b.load[1](
                tile * TPB + local_row, tiled_col
            );
        
        barrier();

        # Matrix multiplication within the tile
        if tiled_row < size and tiled_col < size:
            @parameter
            for k in range(TPB):
                sum += sm_tile_a.load[1](local_row, k) * sm_tile_b.load[1](k, local_col);
        
        barrier();

    # Write out final result
    if tiled_row < size and tiled_col < size:
        output.store[1](tiled_row, tiled_col, sum);


fn main() raises:
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE_TILED * SIZE_TILED).enqueue_fill(0);
        inp1 = ctx.enqueue_create_buffer[dtype](SIZE_TILED * SIZE_TILED).enqueue_fill(0);
        inp2 = ctx.enqueue_create_buffer[dtype](SIZE_TILED * SIZE_TILED).enqueue_fill(0);
        expected = ctx.enqueue_create_host_buffer[dtype](
            SIZE_TILED * SIZE_TILED
        ).enqueue_fill(0);
        with inp1.map_to_host() as inp1_host, inp2.map_to_host() as inp2_host:
            for row in range(SIZE_TILED):
                for col in range(SIZE_TILED):
                    val = row * SIZE_TILED + col;
                    # row major: placing elements row by row
                    inp1_host[row * SIZE_TILED + col] = val;
                    inp2_host[row * SIZE_TILED + col] = Float32(2.0) * val;
            
            # inp1 @ inp2.T
            for i in range(SIZE_TILED):
                for j in range(SIZE_TILED):
                    for k in range(SIZE_TILED):
                        expected[i * SIZE_TILED + j] += (
                            inp1_host[i * SIZE_TILED + k] * inp2_host[k * SIZE_TILED + j]
                        );
            
        out_tensor = LayoutTensor[mut=True, dtype, layout_tiled](out.unsafe_ptr());
        a_tensor = LayoutTensor[mut=False, dtype, layout_tiled](inp1.unsafe_ptr());
        b_tensor = LayoutTensor[mut=False, dtype, layout_tiled](inp2.unsafe_ptr());

        ctx.enqueue_function[matmul_tiled[layout_tiled, SIZE_TILED]](
            out_tensor,
            a_tensor,
            b_tensor,
            grid_dim=BLOCKS_PER_GRID_TILED,
            block_dim=THREADS_PER_BLOCK_TILED,
        );

        ctx.synchronize();

        with out.map_to_host() as out_host:
            print("out:", out_host);
            print("expected:", expected);
            for col in range(SIZE_TILED):
                for row in range(SIZE_TILED):
                    assert_equal(
                        out_host[col * SIZE_TILED + row], expected[col * SIZE_TILED + row]
                    );

            # Verify results match
            var all_match = True;
            for i in range(SIZE_TILED * SIZE_TILED):
                var diff = abs(out_host[i] - expected[i]);
                if diff > 1e-5:
                    print("Mismatch at index", i, ": GPU =", out_host[i], "Expected =", expected[i]);
                    all_match = False;
                    # break  # Remove break to see all mismatches
                    
            if all_match:
                print("\n✅ All Results Match!");
            else:
                print("\n❌ Results Differ!");