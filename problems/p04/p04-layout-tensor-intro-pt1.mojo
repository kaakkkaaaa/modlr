from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

comptime HEIGHT = 2;
comptime WIDTH = 3;
comptime dtype = DType.float32;
comptime layout = Layout.row_major(HEIGHT, WIDTH);

fn kernel[
    dtype: DType, layout: Layout
](tensor: LayoutTensor[dtype, layout, MutAnyOrigin]):
    tensor[0, 0] += 1;

def main():
    ctx = DeviceContext();

    # Create device buffer
    device_buf = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH);
    device_buf.enqueue_fill(0);

    # Create host buffer for printing results
    host_buf = ctx.enqueue_create_host_buffer[dtype](HEIGHT * WIDTH);

    # Wait for buffers to be created
    ctx.synchronize();

    print("Before kernel:");
    print(host_buf);

    # Create tensor from device buffer
    tensor = LayoutTensor[dtype, layout, MutAnyOrigin](device_buf);

    # Run the kernel
    ctx.enqueue_function_checked[kernel[dtype, layout], kernel[dtype, layout]](
        tensor,
        grid_dim=1,
        block_dim=1
    );

    # Copy results from device to host
    ctx.enqueue_copy(dst_buf=host_buf, src_buf=device_buf);

    # Wait for all operations to complete
    ctx.synchronize();

    print("After kernel:");
    print(host_buf);