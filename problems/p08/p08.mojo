from std.memory import Pointer, stack_allocation as mem_stack_allocation
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorEngine, row_major
from layout.tile_tensor import stack_allocation as tensor_stack_allocation
from std.testing import assert_equal


# ANCHOR: add_10_shared
comptime TPB = 4;
comptime SIZE = 8;
comptime BLOCKS_PER_GRID = (2, 1);
comptime THREADS_PER_BLOCK = (TPB, 1);
comptime dtype = DType.float32;
comptime layout = row_major[SIZE]();
comptime LayoutType = type_of(layout);

# 1 : TileTensor
def add_10_shared[
    Engine: TensorEngine,
](
    output: TileTensor[
        mut=True, dtype, LayoutType, MutAnyOrigin, Engine=Engine
    ],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin, Engine=Engine],
    size_dev: Int32,
) where (Engine.element_size == 1):
    var size = Int(size_dev);
    var shared = tensor_stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[TPB](),
    );
    var global_i = block_dim.x * block_idx.x + thread_idx.x;
    var local_i = thread_idx.x;
    if global_i < size:
        shared[local_i] = rebind[Scalar[dtype]](a[global_i]);
    barrier();
    if global_i < size:
        output[global_i] = shared[local_i] + 10;

# 2 : Raw Memory
def add_10_shared_raw_mem(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev);
    var shared = mem_stack_allocation[
        TPB, Scalar[dtype], address_space = AddressSpace.SHARED
    ]();
    var global_i = block_dim.x * block_idx.x + thread_idx.x;
    var local_i = thread_idx.x;
    # local data into shared memory
    if global_i < size:
        shared[unsafe_offset=local_i] = a[unsafe_offset=global_i];
    
    # wait for all threads to complete
    # works within a thread block
    barrier();

    # FILL ME IN (roughly 2 lines)
    if global_i < size:
        output[unsafe_offset=global_i] = shared[unsafe_offset=local_i] + 10.0;

# ANCHOR_END: add_10_shared

def check(buf: DeviceBuffer[dtype], name: String) raises:
    with buf.map_to_host() as out_host:
        print(name, ":", out_host);
        for i in range(SIZE):
            assert_equal(out_host[i], Scalar[dtype](i + 10));
    print(name, "passed ✅");


def main() raises:
    with DeviceContext() as ctx:
        # Shared input: a[i] = i;
        var a = ctx.enqueue_create_buffer[dtype](SIZE);
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i);
        
        # 1) TileTensor version
        var out_tile = ctx.enqueue_create_buffer[dtype](SIZE);
        out_tile.enqueue_fill(0);
        var out_tensor = TileTensor(out_tile, layout);
        var a_tensor = TileTensor(a, layout);
        ctx.enqueue_function[add_10_shared[out_tensor.Engine]](
            out_tensor,
            a_tensor,
            Int32(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        );
        # 2) Raw Memory version
        var out_raw = ctx.enqueue_create_buffer[dtype](SIZE);
        out_raw.enqueue_fill(0);
        ctx.enqueue_function[add_10_shared_raw_mem](
            out_raw,
            a,
            Int32(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        );

        ctx.synchronize();

        print("--------------------------------");
        check(out_tile, "TileTensor");
        print("--------------------------------");
        check(out_raw, "Raw Memory");
        print("--------------------------------");
        print("Puzzle 8 complete ✅");
