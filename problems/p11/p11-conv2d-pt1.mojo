from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from memory import stack_allocation, UnsafePointer
from sys import sizeof
from testing import assert_equal

# 2D Convoluation parameters
alias TILE_SIZE = 16;       # Thread block size (16x16)
alias CONV_SIZE = 3;        # Kernel size (3x3)
alias INPUT_HEIGHT = 64;    # Input image height
alias INPUT_WIDTH = 64;     # Input image width
alias PADDING = 1;          # Zero padding
alias STRIDE = 1;           # Convolution stride
alias dtype = DType.float32;

# Calculate output dimensions
alias OUTPUT_HEIGHT = (INPUT_HEIGHT + 2 * PADDING - CONV_SIZE) // STRIDE + 1;
alias OUTPUT_WIDTH = (INPUT_WIDTH + 2 * PADDING - CONV_SIZE) // STRIDE + 1;

# Grid and block dimensions for 2D execution
alias BLOCKS_PER_GRID = (
    (OUTPUT_WIDTH + TILE_SIZE - 1) // TILE_SIZE,    # Grid width
    (OUTPUT_HEIGHT + TILE_SIZE - 1) // TILE_SIZE    # Grid height
);
alias THREADS_PER_BLOCK = (TILE_SIZE, TILE_SIZE);

fn format_float(
    val: Float32,
    precision: Int = 4,
) -> String:
    """Format a float to string with specified precision."""
    return String(val);

# 2D Tensor struct to hold image data
struct Tensor2D:
    var data: HostBuffer[dtype]
    var height: Int
    var width: Int

    fn __init__(out self, height: Int, width: Int, ctx: DeviceContext) raises:
        self.height = height;
        self.width = width;
        self.data = ctx.enqueue_create_host_buffer[dtype](height * width);
        ctx.synchronize();
    
    fn __init__(out self, data: HostBuffer[dtype], height: Int, width: Int):
        self.height = height;
        self.width = width;
        self.data = data;
    
    fn get(self, row: Int, col: Int) -> Float32:
        """Get element at (row, col) with bounds checking."""
        if row >= 0 and row < self.height and col >= 0 and col < self.width:
            return self.data[row * self.width + col]
        return Float32(0.0)     # Zero padding for out-of-bounds
    
    # Improved to_string function to match PyTorch format
    fn to_string(self, max_rows: Int = 5, max_cols: Int = 5) raises -> String:
        var s = String("tensor([")
        # Handle empty matrix
        if self.height == 0 or self.width == 0:
            s += "])"
            return s
        
        # Determine how many rows to show before ellipsis
        var rows_before_ellipsis = min(max_rows - 1, self.height)
        
        # Process rows before ellipsis
        for row in range(rows_before_ellipsis):
            if row > 0:
                s += ",\n "
            s += "["
            
            # Process columns for this row
            if self.height <= max_cols:
                # Show all columns
                for col in range(self.width):
                    if col > 0:
                        s += ", "
                    var val = self.data[row * self.width + col]
                    s += format_float(val)  # Now calls the standalone function
            else:
                # Show first max_cols-1 elements, ellipsis, and last element
                for col in range(max_cols - 1):
                    if col > 0:
                        s += ", "
                    var val = self.data[row * self.width + col]
                    s += format_float(val)  # Now calls the standalone function
                s += ", ..."
                # Add last element
                var last_val = self.data[row * self.width + self.width - 1]
                s += String(", ") + format_float(last_val)  # Now calls the standalone function
            s += "]"
        
        # Add ellipsis row if there are more rows
        if self.height > max_rows - 1:
            s += ",\n ..."
            # Add last row
            s += ",\n ["
            
            # Process columns for last row
            if self.width <= max_cols:
                # Show all columns
                for col in range(self.width):
                    if col > 0:
                        s += ", "
                    var val = self.data[(self.height - 1) * self.width + col]
                    s += format_float(val)  # Now calls the standalone function
            else:
                # Show first max_cols-1 elements, ellipsis, and last element
                for col in range(max_cols - 1):
                    if col > 0:
                        s += ", "
                    var val = self.data[(self.height - 1) * self.width + col]
                    s += format_float(val)  # Now calls the standalone function
                s += ", ..."
                # Add last element
                var last_val = self.data[(self.height - 1) * self.width + self.width - 1]
                s += String(", ") + format_float(last_val)  # Now calls the standalone function
            s += "]"
        
        s += "])"
        return s

# GPU kernel for 2D Convolution
fn conv2d_kernel(
    output: UnsafePointer[SIMD[dtype, 1]],
    input: UnsafePointer[SIMD[dtype, 1]],
    kernel: UnsafePointer[SIMD[dtype, 1]],
    input_height: Int,
    input_width: Int,
    output_height: Int,
    output_width: Int,
    kernel_size: Int,
    padding: Int,
    stride: Int
):
    # Calculate global thread indices (output coordinates)
    var output_col = block_dim.x * block_idx.x + thread_idx.x;
    var output_row = block_dim.y * block_idx.y + thread_idx.y;

    # Calculate local thread indices
    var local_col = thread_idx.x;
    var local_row = thread_idx.y;

    # Allocate shared memory for input tile
    # Add padding around the tile to handle convolution boundaries
    var shared_size = TILE_SIZE + 2 * PADDING;
    var shared_input = stack_allocation[
        (TILE_SIZE + 2 * PADDING) * (TILE_SIZE + 2 * PADDING),
        SIMD[dtype, 1],
        address_space=AddressSpace.SHARED
    ]();

    # Allocate shared memory for kernel
    var shared_kernel = stack_allocation[
        CONV_SIZE * CONV_SIZE,
        SIMD[dtype, 1],
        address_space=AddressSpace.SHARED
    ]();

    # Load kernel into shared memory (once per block)
    if local_row < kernel_size and local_col < kernel_size:
        shared_kernel[local_row * kernel_size + local_col] = kernel[local_row * kernel_size + local_col];
    
    # Calculate input coordinates (main tile loading)
    var input_row = block_idx.y * TILE_SIZE + local_row;
    var input_col = block_idx.x * TILE_SIZE + local_col;

    # Calculate shared memory position
    var shared_row = local_row + PADDING;   # Offset by padding
    var shared_col = local_col + PADDING;   # Offset by padding
    var shared_idx = shared_row * shared_size + shared_col;

    # Load main tile into shared memory
    if input_row >= 0 and input_row < input_height and input_col >= 0 and input_col < input_width:
        shared_input[shared_idx] = input[input_row * input_width + input_col];
    else:
        shared_input[shared_idx] = Float32(0.0);    # Zero padding
    
    # Load boundary elements
    # Top boundary
    if local_row < PADDING:
        var boundary_input_row = input_row - PADDING;
        var boundary_shared_idx = local_row * shared_size + shared_col;
        if boundary_input_row >= 0 and boundary_input_row < input_height and input_col >= 0 and input_col < input_width:
            shared_input[boundary_shared_idx] = input[boundary_input_row * input_width + input_col];
        else:
            shared_input[boundary_shared_idx] = Float32(0.0);
        
    # Bottom boundary
    if local_row >= TILE_SIZE - PADDING:
        var boundary_input_row = input_row + PADDING;
        var boundary_shared_idx = (shared_row + PADDING) * shared_size + shared_col;
        if boundary_input_row >= 0 and boundary_input_row < input_height and input_col >= 0 and input_col < input_width:
            shared_input[boundary_shared_idx] = input[boundary_input_row * input_width + input_col];
        else:
            shared_input[boundary_shared_idx] = Float32(0.0);
    
    # Left boundary
    if local_col < PADDING:
        var boundary_input_col = input_col - PADDING;
        var boundary_shared_idx = shared_row * shared_size + local_col;
        if input_row >= 0 and input_row < input_height and boundary_input_col >= 0 and boundary_input_col < input_width:
            shared_input[boundary_shared_idx] = input[input_row * input_width + boundary_input_col];
        else:
            shared_input[boundary_shared_idx] = Float32(0.0);
    
    # Right boundary
    if local_col >= TILE_SIZE - PADDING:
        var boundary_input_col = input_col + PADDING;
        var boundary_shared_idx = shared_row * shared_size + (shared_col + PADDING);
        if input_row >= 0 and input_row < input_height and boundary_input_col >= 0 and boundary_input_col < input_width:
            shared_input[boundary_shared_idx] = input[input_row * input_width + boundary_input_col];
        else:
            shared_input[boundary_shared_idx] = Float32(0.0);
    
    # Load corner elements (diagonal corners of halo region)
    # Top-left corner
    if local_row < PADDING and local_col < PADDING:
        var corner_input_row = input_row - PADDING
        var corner_input_col = input_col - PADDING
        var corner_shared_idx = local_row * shared_size + local_col
        if corner_input_row >= 0 and corner_input_row < input_height and corner_input_col >= 0 and corner_input_col < input_width:
            shared_input[corner_shared_idx] = input[corner_input_row * input_width + corner_input_col]
        else:
            shared_input[corner_shared_idx] = Float32(0.0)
    
    # Top-right corner
    if local_row < PADDING and local_col >= TILE_SIZE - PADDING:
        var corner_input_row = input_row - PADDING
        var corner_input_col = input_col + PADDING
        var corner_shared_idx = local_row * shared_size + (shared_col + PADDING)
        if corner_input_row >= 0 and corner_input_row < input_height and corner_input_col >= 0 and corner_input_col < input_width:
            shared_input[corner_shared_idx] = input[corner_input_row * input_width + corner_input_col]
        else:
            shared_input[corner_shared_idx] = Float32(0.0)
    
    # Bottom-left corner
    if local_row >= TILE_SIZE - PADDING and local_col < PADDING:
        var corner_input_row = input_row + PADDING
        var corner_input_col = input_col - PADDING
        var corner_shared_idx = (shared_row + PADDING) * shared_size + local_col
        if corner_input_row >= 0 and corner_input_row < input_height and corner_input_col >= 0 and corner_input_col < input_width:
            shared_input[corner_shared_idx] = input[corner_input_row * input_width + corner_input_col]
        else:
            shared_input[corner_shared_idx] = Float32(0.0)
    
    # Bottom-right corner
    if local_row >= TILE_SIZE - PADDING and local_col >= TILE_SIZE - PADDING:
        var corner_input_row = input_row + PADDING
        var corner_input_col = input_col + PADDING
        var corner_shared_idx = (shared_row + PADDING) * shared_size + (shared_col + PADDING)
        if corner_input_row >= 0 and corner_input_row < input_height and corner_input_col >= 0 and corner_input_col < input_width:
            shared_input[corner_shared_idx] = input[corner_input_row * input_width + corner_input_col]
        else:
            shared_input[corner_shared_idx] = Float32(0.0)

    # Synchronize all threads before computation
    barrier();

    # Compute convolution for this thread's output element
    if output_row < output_height and output_col < output_width:
        var sum = SIMD[dtype, 1](0.0);

        # Perform 2D convolution
        for kr in range(kernel_size):
            for kc in range(kernel_size):
                # Calculate position in shared memory for convolution window
                var input_shared_row = shared_row + kr - PADDING;
                var input_shared_col = shared_col + kc - PADDING;
                var input_shared_idx = input_shared_row * shared_size + input_shared_col;
                var kernel_idx = kr * kernel_size + kc;

                if (input_shared_row >= 0 and input_shared_row < shared_size and
                    input_shared_col >= 0 and input_shared_col < shared_size):
                    sum += shared_input[input_shared_idx] * shared_kernel[kernel_idx]

        # Write result to output
        output[output_row * output_width + output_col] = sum;
    

# CPU 2D Convolution for verification
fn cpu_conv2d(
    output: HostBuffer[dtype],
    input: HostBuffer[dtype],
    kernel: HostBuffer[dtype],
    input_height: Int,
    input_width: Int,
    output_height: Int,
    output_width: Int,
    kernel_size: Int,
    padding: Int,
    stride: Int
):
    for out_row in range(output_height):
        for out_col in range(output_width):
            var sum = SIMD[dtype, 1](0.0);

            for kr in range(kernel_size):
                for kc in range(kernel_size):
                    var in_row = out_row * stride - padding + kr;
                    var in_col = out_col * stride - padding + kc;

                    # Check bounds and apply zero padding
                    if in_row >= 0 and in_row < input_height and in_col >= 0 and in_col < input_width:
                        var input_val = input[in_row * input_width + in_col];
                        var kernel_val = kernel[kr * kernel_size + kc];
                        sum += input_val * kernel_val;
            
            output[out_row * output_width + out_col] = sum;

fn main() raises:
    with DeviceContext() as ctx:
        # Create host buffers
        var input_host = ctx.enqueue_create_host_buffer[dtype](INPUT_HEIGHT * INPUT_WIDTH);
        var kernel_host = ctx.enqueue_create_host_buffer[dtype](CONV_SIZE * CONV_SIZE);
        var output_host = ctx.enqueue_create_host_buffer[dtype](OUTPUT_HEIGHT * OUTPUT_WIDTH);
        var output_cpu = ctx.enqueue_create_host_buffer[dtype](OUTPUT_HEIGHT * OUTPUT_WIDTH);

        # Synchronize to ensure buffers are created
        ctx.synchronize();

        # Fill input with sample data (simple pattern)
        for i in range(INPUT_HEIGHT):
            for j in range(INPUT_WIDTH):
                # Create a simple pattern: checkerboard-like
                var value = Float32((i + j) % 10);
                input_host[i * INPUT_WIDTH + j] = value;
        
        # Create a simple edge detection kernel
        var edge_kernel = [
            -1.0, -1.0, -1.0,
            -1.0,  8.0, -1.0,
            -1.0, -1.0, -1.0
        ];

        for i in range(CONV_SIZE * CONV_SIZE):
            kernel_host[i] = Float32(edge_kernel[i]);
        
        # Initialize output arrays
        for i in range(OUTPUT_HEIGHT * OUTPUT_WIDTH):
            output_host[i] = Float32(0.0);
            output_cpu[i] = Float32(0.0);

        # Create tensor objects for pretty printing
        var input_tensor = Tensor2D(input_host, INPUT_HEIGHT, INPUT_WIDTH);
        var kernel_tensor = Tensor2D(kernel_host, CONV_SIZE, CONV_SIZE);

        # Print input arrays
        print(String("Input Image ({}x{}):").format(INPUT_HEIGHT, INPUT_WIDTH));
        print(input_tensor.to_string());

        print(String("\nConvolution Kernel ({}x{})").format(CONV_SIZE, CONV_SIZE));
        print(kernel_tensor.to_string());

        # Create device buffers
        var input_device = ctx.enqueue_create_buffer[dtype](INPUT_HEIGHT * INPUT_WIDTH);
        var kernel_device = ctx.enqueue_create_buffer[dtype](CONV_SIZE * CONV_SIZE);
        var output_device = ctx.enqueue_create_buffer[dtype](OUTPUT_HEIGHT * OUTPUT_WIDTH);

        # Copy data to device
        ctx.enqueue_copy(input_device, input_host);
        ctx.enqueue_copy(kernel_device, kernel_host);
        ctx.enqueue_memset(output_device, Float32(0.0));

        print("\nLaunching GPU kernel with:");
        print("- Grid dimensions:", BLOCKS_PER_GRID[0], "x", BLOCKS_PER_GRID[1]);
        print("- Block dimensions:", THREADS_PER_BLOCK[0], "x", THREADS_PER_BLOCK[1]);
        print("- Output size:", OUTPUT_HEIGHT, "x", OUTPUT_WIDTH);

        # Launch GPU kernel
        ctx.enqueue_function[conv2d_kernel](
            output_device.unsafe_ptr(),
            input_device.unsafe_ptr(),
            kernel_device.unsafe_ptr(),
            INPUT_HEIGHT,
            INPUT_WIDTH,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
            CONV_SIZE,
            PADDING,
            STRIDE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        );

        # Copy result back to host
        ctx.enqueue_copy(output_host, output_device);

        # Synchronize to ensure all operations complete
        ctx.synchronize();

        # Compute CPU version for verification
        cpu_conv2d(
            output_cpu,
            input_host,
            kernel_host,
            INPUT_HEIGHT,
            INPUT_WIDTH,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
            CONV_SIZE,
            PADDING,
            STRIDE
        );
        
        # Create tensor objects for results
        var output_gpu_tensor = Tensor2D(output_host, OUTPUT_HEIGHT, OUTPUT_WIDTH);
        var output_cpu_tensor = Tensor2D(output_cpu, OUTPUT_HEIGHT, OUTPUT_WIDTH);

        print(String("\nGPU Result ({}x{})").format(OUTPUT_HEIGHT, OUTPUT_WIDTH));
        print(output_gpu_tensor.to_string());

        print(String("\nCPU Result ({}x{})").format(OUTPUT_HEIGHT, OUTPUT_WIDTH));
        print(output_cpu_tensor.to_string());


        # Verify results match
        var tolerance = Float32(1e-5);
        var matches = True;
        var max_error = Float32(0.0);
        
        for i in range(OUTPUT_HEIGHT * OUTPUT_WIDTH):
            var error = abs(output_host[i] - output_cpu[i]);
            max_error = max(max_error, error);
            if error > tolerance:
                matches = False;
                var row = i // OUTPUT_WIDTH;
                var col = i % OUTPUT_WIDTH;
                print("Mismatch at (", row, ",", col, "):", output_host[i], "vs", output_cpu[i]);
                break;
        
        if matches:
            print("\n✅ GPU and CPU results match!");
            print("Maximum error:", format_float(max_error));
        else:
            print("\n❌ GPU and CPU results differ!");
            print("Maximum error:", format_float(max_error));