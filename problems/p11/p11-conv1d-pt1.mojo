from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from memory import stack_allocation, UnsafePointer
from sys import sizeof
from testing import assert_equal

# Convolution parameters
alias TPB = 32;
alias SIZE = 1024;
alias CONV = 16;
alias BLOCKS_PER_GRID = (32, 1);
alias THREADS_PER_BLOCK = (TPB, 1);
alias dtype = DType.float32;

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

# GPU kernel for 1D Convolution
fn conv1d_kernel(
    output: UnsafePointer[SIMD[dtype, 1]],
    input: UnsafePointer[SIMD[dtype, 1]],
    conv: UnsafePointer[SIMD[dtype, 1]],
    input_size: Int,
    conv_size: Int
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x;
    var local_i = thread_idx.x;

    # Allocate shared memory with overlap for boundary handling
    var shared_input = stack_allocation[
        TPB + CONV - 1,
        SIMD[dtype, 1],
        address_space=AddressSpace.SHARED
    ]();

    var shared_kernel = stack_allocation[
        CONV,
        SIMD[dtype, 1],
        address_space=AddressSpace.SHARED
    ]();

    # Load input data into shared memory
    if global_i < input_size:
        shared_input[local_i] = input[global_i];
    else:
        shared_input[local_i] = 0.0;    # Zero padding
    
    # Load overlap data (boundary threads only)
    if local_i < conv_size - 1:
        var next_idx = global_i + TPB;
        if next_idx < input_size:
            shared_input[TPB + local_i] = input[next_idx];
        else:
            shared_input[TPB + local_i] = 0.0;
    
    # Load kernel data into shared memory
    if local_i < conv_size:
        shared_kernel[local_i] = conv[local_i];
    
    # Synchronous all threads
    barrier();

    # Compute convolution for this thread's output element
    if global_i < input_size:
        var local_sum = SIMD[dtype, 1](0.0);

        # Perform convolution
        for j in range(conv_size):
            var input_idx = local_i + j;
            if input_idx < (TPB + CONV - 1):
                local_sum += shared_input[input_idx] * shared_kernel[j];
        
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
            input_host[i] = Float32(i + 1); # [1, 2, 3, 4, 5, 6]
        
        for i in range(CONV):
            conv_host[i] = Float32(1) / Float32(CONV);
        
        # Initialize output arrays
        for i in range(SIZE):
            output_host[i] = Float32(0.0);
            output_cpu[i] = Float32(0.0);
        
        # Create array objects for pretty printing
        var inp_arr = Tensor1D(input_host, SIZE);
        var conv_arr = Tensor1D(conv_host, CONV);

        # Print input arrays
        print("Input Array:");
        print(inp_arr.to_string());

        print("\nConv Array:");
        print(conv_arr.to_string());

        # Create device buffers
        var inp_device = ctx.enqueue_create_buffer[dtype](SIZE);
        var conv_device = ctx.enqueue_create_buffer[dtype](CONV);
        var out_device = ctx.enqueue_create_buffer[dtype](SIZE);

        # Copy data to device
        ctx.enqueue_copy(inp_device, input_host);
        ctx.enqueue_copy(conv_device, conv_host);
        ctx.enqueue_memset(out_device, Float32(0.0));

        # Launch GPU kernel
        ctx.enqueue_function[conv1d_kernel](
            out_device.unsafe_ptr(),
            inp_device.unsafe_ptr(),
            conv_device.unsafe_ptr(),
            SIZE,
            CONV,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        );

        # Copy result back to host
        ctx.enqueue_copy(output_host, out_device);

        # Synchronize to ensure all operations complete
        ctx.synchronize();

        # Compute CPU version for verification
        cpu_conv_1d(output_cpu, input_host, conv_host, SIZE, CONV);

        # Create array objects for results
        var output_gpu_array = Tensor1D(output_host, SIZE);
        var output_cpu_array = Tensor1D(output_cpu, SIZE);

        print("\nGPU Result:");
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
        # if matches:
        #     print("\nDetailed Comparison:");
        #     for i in range(SIZE):
        #         print("✅ Index", i, "- GPU:", format_float(output_host[i]),
        #         "CPU:", format_float(output_cpu[i]),
        #         "Diff:", format_float(abs(output_host[i] - output_cpu[i])));