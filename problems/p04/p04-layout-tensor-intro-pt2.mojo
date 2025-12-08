from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from memory import UnsafePointer

comptime HEIGHT = 2;
comptime WIDTH = 3;
comptime dtype = DType.float32;
comptime layout = Layout.row_major(HEIGHT, WIDTH);

fn kernel(
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]
):
    var tensor = LayoutTensor[dtype, layout](ptr);
    tensor[0, 0] += 1;

fn main() raises:
    var ctx = DeviceContext();

    # Create device buffer
    var device_buf = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH);
    device_buf.enqueue_fill(0);

    # Create host buffer for results
    var host_buf = ctx.enqueue_create_host_buffer[dtype](HEIGHT * WIDTH);

    # Wait for buffers to be created
    ctx.synchronize();

    print("Before:");
    print(host_buf);

    # Run the kernel
    ctx.enqueue_function_checked[kernel, kernel](
        device_buf,
        grid_dim=1,
        block_dim=1
    );

    # Copy results from device to host
    ctx.enqueue_copy(dst_buf=host_buf, src_buf=device_buf);

    # Wait for all operations to complete
    ctx.synchronize();

    print("After:");
    print(host_buf);