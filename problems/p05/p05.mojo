from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorEngine
from layout.tile_layout import row_major
from std.testing import assert_equal

# ANCHOR: broadcast_add
comptime SIZE = 2;
comptime BLOCKS_PER_GRID = 1;
comptime THREADS_PER_BLOCK = (3, 3);
comptime dtype = DType.float32;
comptime out_layout = row_major[SIZE, SIZE]();
comptime a_layout = row_major[1, SIZE]();
comptime b_layout = row_major[SIZE, 1]();
comptime OutLayout = type_of(out_layout);
comptime ALayout = type_of(a_layout);
comptime BLayout = type_of(b_layout);


def broadcast_add[
    Engine: TensorEngine,
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin, Engine=Engine],
    a: TileTensor[mut=False, dtype, ALayout, ImmutAnyOrigin, Engine=Engine],
    b: TileTensor[mut=False, dtype, BLayout, ImmutAnyOrigin, Engine=Engine],
    size_dev: Int32,
) where (Engine.element_size == 1):
    var size = Int(size_dev);
    var row = thread_idx.y;
    var col = thread_idx.x;
    # FILL ME IN (roughly 2 lines)
    if row < size and col < size:
        output[row, col] = a[0, col] + b[row, 0];


# ANCHOR_END: broadcast_add
def main() raises:
    with DeviceContext() as ctx:
        print(t"Running on: {ctx.name()} ({ctx.api()}, device {ctx.id()})");
        var out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE);
        out.enqueue_fill(0);
        var out_tensor = TileTensor(out, out_layout);
        print("out shape:", out_tensor.dim[0](), "x", out_tensor.dim[1]());

        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE);
        expected.enqueue_fill(0);
        var expected_tensor = TileTensor(expected, out_layout);

        var a = ctx.enqueue_create_buffer[dtype](SIZE);
        a.enqueue_fill(0);
        var b = ctx.enqueue_create_buffer[dtype](SIZE);
        b.enqueue_fill(0);
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i + 1);
                b_host[i] = Scalar[dtype](i * 10);
            
            for i in range(SIZE):
                for j in range(SIZE):
                    expected_tensor[i, j] = a_host[j] + b_host[i];
        
        var a_tensor = TileTensor(a, a_layout);
        var b_tensor = TileTensor(b, b_layout);

        ctx.enqueue_function[broadcast_add[out_tensor.Engine]](
            out_tensor,
            a_tensor,
            b_tensor,
            Int32(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        );

        ctx.synchronize();

        with out.map_to_host() as out_host:
            print("out:", out_host);
            print("expected:", expected);
            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(out_host[i * SIZE + j], expected[i * SIZE + j]);
        
        print("Puzzle 05 complete ✅");

