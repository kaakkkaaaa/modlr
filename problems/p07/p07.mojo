from std.memory import Pointer
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorEngine, row_major
from std.testing import assert_equal

# ANCHOR: add_10_blocks_2d
comptime SIZE = 5;
comptime BLOCKS_PER_GRID = (2, 2);
comptime THREADS_PER_BLOCK = (3, 3);
comptime dtype = DType.float32;
comptime out_layout = row_major[SIZE, SIZE]();
comptime a_layout = row_major[SIZE, SIZE]();
comptime OutLayout = type_of(out_layout);
comptime ALayout = type_of(a_layout);


def add_10_blocks_2d[
    Engine: TensorEngine
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin, Engine=Engine],
    a: TileTensor[mut=False, dtype, ALayout, ImmutAnyOrigin, Engine=Engine],
    size_dev: Int32,
) where (Engine.element_size == 1):
    var size = Int(size_dev);
    var row = block_dim.y * block_idx.y + thread_idx.y;
    var col = block_dim.x * block_idx.x + thread_idx.x;
    # FILL ME IN (roughly 2 lines)
    if row < size and col < size:
        output[row, col] = a[row, col] + 10.0;

def add_10_blocks_2d_raw_mem(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev);
    var row = block_dim.y * block_idx.y + thread_idx.y;
    var col = block_dim.x * block_idx.x + thread_idx.x;
    # FILL ME IN (roughly 2 lines)
    if row < size and col < size:
        output[unsafe_offset=(row * size + col)] = a[unsafe_offset=(row * size + col)] + 10.0;

# ANCHOR_END: add_10_blocks_2d

def check(buf: DeviceBuffer[dtype], name: String) raises:
    with buf.map_to_host() as out_host:
        print(name, ":", out_host);
        for i in range(SIZE):
            for j in range(SIZE):
                var idx = i * SIZE + j;
                assert_equal(out_host[idx], Scalar[dtype](idx + 10));
    print("✅", name, "passed");

def main() raises:
    with DeviceContext() as ctx:
        # Shared input: a[i, j] = i * SIZE + j (0..24)
        var a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE);
        with a.map_to_host() as a_host:
            for i in range(SIZE * SIZE):
                a_host[i] = Scalar[dtype](i);
        
        # TileTensor
        var output_tile = ctx.enqueue_create_buffer[dtype](SIZE * SIZE);
        output_tile.enqueue_fill(0);
        var output_tensor = TileTensor(output_tile, out_layout);
        var a_tensor = TileTensor(a, a_layout);
        comptime tile_tensor_kernel = add_10_blocks_2d[type_of(output_tensor).Engine];

        ctx.enqueue_function[tile_tensor_kernel](
            output_tensor,
            a_tensor,
            Int32(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        );

        # Raw Memory
        var output_raw = ctx.enqueue_create_buffer[dtype](SIZE * SIZE);
        output_raw.enqueue_fill(0);
        ctx.enqueue_function[add_10_blocks_2d_raw_mem](
            output_raw,
            a,
            Int32(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        );
        ctx.synchronize();

        print("--------------------------------");
        check(output_tile, "TileTensor");
        print("--------------------------------");
        check(output_raw, "Raw Memory");
        print("--------------------------------");
        print("Puzzle 07 complete ✅");
