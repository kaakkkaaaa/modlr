from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof
from testing import assert_equal

# Convolutional Parameters
alias TPB = 8;
alias SIZE = 15;
alias CONV = 3;
alias BLOCKS_PER_GRID = (2, 1);
alias THREADS_PER_BLOCK = (TPB, 1);
alias dtype = DType.float32;

alias in_layout = Layout.row_major(SIZE);
alias out_layout = Layout.row_major(SIZE);
alias conv_layout = Layout.row_major(CONV);

fn format_float(
    val: Float32,
    precision: Int = 4
) -> String:
    """Format a float to string with specified precision."""
    return String(val);

# Array struct to hold data and dimension (similar to Matrix)
struct Tensor1D:
    var data: HostBuffer[dtype]
    var size: Int

    fn __init__(out self, size: Int, ctx: DeviceContext) raises:
        self.size = size;
        self.data = ctx.enqueue_create_host_buffer[dtype](size);
        ctx.synchronize();
    
    fn __init__(out self, data: HostBuffer[dtype], size: Int):
        self.size = size;
        self.data = data;
    
    fn to_string(self, max_elements: Int = 10) raises -> String:
        var s = String("Tensor([");
        if self.size == 0:
            s += "])";
            return s
        
        if self.size <= max_elements:
            # Show all elements
            for i in range(self.size):
                if i > 0:
                    s += ", ";
                s += format_float(self.data[i]);
        else:
            # Show first few elements, ellipsis, and last element
            var show_count = max_elements - 1;
            for i in range(show_count):
                if i > 0:
                    s += ", ";
                s += format_float(self.data[i]);
            s += ", ..., " + format_float(self.data[self.size - 1]);
        
        s += "])";
        return s

# GPU kernel for 1D convolution using LayoutTensor
fn conv1d_kernel[
    in_layout: Layout,
    out_layout: Layout,
    conv_layout: Layout
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    input: LayoutTensor[mut=False, dtype, in_layout],
    conv: LayoutTensor[mut=False, dtype, conv_layout],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x;
    var local_i = thread_idx.x;

    # Allocate shared memory with overlap for boundary handling
    # Extended size to handle block boundaries: TPB + CONV - 1
    var shared_input = tb[dtype]().row_major[TPB + CONV - 1]().shared().alloc();
    var shared_conv = tb[dtype]().row_major[CONV]().shared().alloc();

    # Load input data into shared memory
    if global_i < input.dim(0):         # Fixed: use dim(0) for 1D tensor
        shared_input[local_i] = input[global_i];
    else:
        shared_input[local_i] = 0.0;    # Zero padding

    # Load overlap data (boundary threads only)
    if local_i < CONV - 1:
        var next_idx = global_i + TPB;
        if next_idx < input.dim(0):
            shared_input[TPB + local_i] = input[next_idx];
        else:
            shared_input[TPB + local_i] = 0.0;  # Zero padding
    
    # Load kernel data into shared memory
    if local_i < CONV:
        shared_conv[local_i] = conv[local_i];
    
    # Synchronize all threads
    barrier();

    # Compute convolution for this thread's output element
    if global_i < output.dim(0):
        var local_sum: output.element_type = 0.0;

        # Perform convolution with compile-time loop unrolling
        @parameter
        for j in range(CONV):
            var input_idx = local_i + j;
            if input_idx < (TPB + CONV - 1):
                local_sum += shared_input[input_idx] * shared_conv[j];
        
        output[global_i] = local_sum;

# CPU 1D Convolution for verification
fn cpu_conv_1d(
    output: HostBuffer[dtype],
    input: HostBuffer[dtype],
    conv: HostBuffer[dtype],
    input_size: Int,
    conv_size: Int
):
    for i in range(input_size):
        var sum = SIMD[dtype, 1](0.0);
        for j in range(conv_size):
            if i + j < input_size:
                sum += input[i + j] * conv[j];
        output[i] = sum;

fn main() raises:
    with DeviceContext() as ctx:
        # Create host buffers
        var input_host = ctx.enqueue_create_host_buffer[dtype](SIZE);
        var conv_host = ctx.enqueue_create_host_buffer[dtype](CONV);
        var output_host = ctx.enqueue_create_host_buffer[dtype](SIZE);
        var output_cpu = ctx.enqueue_create_host_buffer[dtype](SIZE);

        # Synchronize to ensure buffers are created
        ctx.synchronize();

        # Fill input and kernel with sample data
        for i in range(SIZE):
            input_host[i] = Float32(i + 1);
        
        for i in range(CONV):
            conv_host[i] = Float32(1) / Float32(CONV);
        
        # Initialize output arrays
        for i in range(SIZE):
            output_host[i] = Float32(0.0);
            output_cpu[i] = Float32(0.0);
        
        # Create array objects for pretty printing
        var input_array = Tensor1D(input_host, SIZE);
        var conv_array = Tensor1D(conv_host, SIZE);

        # Print input arrays
        print("Input Array:");
        print(input_array.to_string());

        print("\nConv Array:");
        print(conv_array.to_string());

        # Create device buffers
        var input_device = ctx.enqueue_create_buffer[dtype](SIZE);
        var conv_device = ctx.enqueue_create_buffer[dtype](CONV);
        var output_device = ctx.enqueue_create_buffer[dtype](SIZE);

        # Copy data to device
        ctx.enqueue_copy(input_device, input_host);
        ctx.enqueue_copy(conv_device, conv_host);
        ctx.enqueue_memset(output_device, Float32(0.0));

        # Create LayoutTensor wrappers for device buffers
        var input_tensor = LayoutTensor[mut=False, dtype, in_layout](
            input_device.unsafe_ptr()
        );
        var conv_tensor = LayoutTensor[mut=False, dtype, conv_layout](
            conv_device.unsafe_ptr()
        );
        var output_tensor = LayoutTensor[mut=True, dtype, out_layout](
            output_device.unsafe_ptr()
        );

        # Launch GPU kernel with LayoutTensor
        ctx.enqueue_function[
            conv1d_kernel[in_layout, out_layout, conv_layout]
        ](
            output_tensor,
            input_tensor,
            conv_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        );

        # Copy result back to host
        ctx.enqueue_copy(output_host, output_device);

        # Synchronize to ensure all operations complete
        ctx.synchronize();

        # Compute CPU version for verification
        cpu_conv_1d(output_cpu, input_host, conv_host, SIZE, CONV);

        # Create array objects for results
        var output_gpu_array = Tensor1D(output_host, SIZE);
        var output_cpu_array = Tensor1D(output_cpu, SIZE);

        print("\nGPU Result (LayoutTensor):");
        print(output_gpu_array.to_string());

        print("\nCPU Result:");
        print(output_cpu_array.to_string());

        # Verify results match
        var tolerance = Float32(1e-5);
        var matches = True;
        for i in range(SIZE):
            if abs(output_host[i] - output_cpu[i]) > tolerance:
                matches = False;
                print("Mismatch at index", i, ":", output_host[i], "vs", output_cpu[i]);
                break;
        
        if matches:
            print("\n✅ GPU and CPU results match!");
        else:
            print("\n❌ GPU and CPU results differ!");

        # Additional detailed verification
        if matches:
            print("\nDetailed comparison:");
            for i in range(SIZE):
                print("✅ Index", i, "- GPU:", format_float(output_host[i]), 
                    "CPU:", format_float(output_cpu[i]), 
                    "Diff:", format_float(abs(output_host[i] - output_cpu[i])));
            print("\n✅ GPU and CPU results match!");