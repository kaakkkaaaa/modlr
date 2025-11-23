from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_equal

alias TPB = 3;
alias SIZE = 2;
alias BLOCKS_PER_GRID = (1, 1);
alias THREADS_PER_BLOCK = (TPB, TPB);
alias dtype = DType.float32;
alias layout = Layout.row_major(SIZE, SIZE);

fn naive_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    row = block_dim.y * block_idx.y + thread_idx.y;
    col = block_dim.x * block_idx.x + thread_idx.x;
    # FILL ME IN (roughly 6 lines)

    if row < size and col < size:
        var sum = SIMD[dtype, 1](0.0);

        @parameter
        for k in range(SIZE):
            sum += a.load[1](row, k) * b.load[1](k, col);
        
        output.store[1](row, col, sum);

fn single_block_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    row = block_dim.y * block_idx.y + thread_idx.y;
    col = block_dim.x * block_idx.x + thread_idx.x;
    local_row = thread_idx.y;
    local_col = thread_idx.x;

    sm_a = tb[dtype]().row_major[SIZE, SIZE]().shared().alloc();
    sm_b = tb[dtype]().row_major[SIZE, SIZE]().shared().alloc();

    if row < size and col < size:
        sm_a[local_row, local_col] = a[row, col];
        sm_b[local_row, local_col] = b[row, col];

        var sum = SIMD[dtype, 1](0.0);

        @parameter
        for k in range(SIZE):
            sum += sm_a.load[1](local_row, k) * b.load[1](k, local_col);
        
        output.store[1](row, col, sum)
    

fn main() raises:
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0);
        inp1 = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0);
        inp2 = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0);
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

        with out.map_to_host() as out_host:
            print("out:", out_host);
            print("expected:", expected);
            for col in range(SIZE):
                for row in range(SIZE):
                    assert_equal(
                        out_host[col * SIZE + row], expected[col * SIZE + row]
                    );