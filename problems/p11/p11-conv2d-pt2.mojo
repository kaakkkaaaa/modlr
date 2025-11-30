from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from memory import stack_allocation, UnsafePointer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof
from testing import assert_equal

# 2D Convolution parameters
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
alias THREADS_PER_BLOCK = (TILE_SIZE, TILE_SIZE);   # 16x16 thread block

# Define layouts for tensors
alias input_layout = Layout.row_major(INPUT_HEIGHT, INPUT_WIDTH);
alias output_layout = Layout.row_major(OUTPUT_HEIGHT, OUTPUT_WIDTH);
alias kernel_layout = Layout.row_major(CONV_SIZE, CONV_SIZE);

# Type aliases for LayoutTensors
alias InputTensor = LayoutTensor[dtype, input_layout, MutableAnyOrigin];
alias OutputTensor = LayoutTensor[dtype, output_layout, MutableAnyOrigin];
alias KernelTensor = LayoutTensor[dtype, kernel_layout, MutableAnyOrigin];

fn format_float(
    val: Float32,
    precision: Int = 4
) -> String:
    """Format a float to string with specified precision."""
    return String(val);

# 2D Tensor struct to hold image data
struct Tensor2D:
    var data: HostBuffer[dtype];
    var height: Int;
    var width: Int;

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
            return self.data[row * self.width + col];
        return Float32(0.0);    # Zero padding for out-of-bounds
    
    fn set(self, row: Int, col: Int, value: Float32):
        """Set element at (row, col)."""
        if row >= 0 and row < self.height and col >= 0 and col < self.width:
            self.data[row * self.width + col] = value;
    
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

# GPU kernel for 2D Convolution using LayoutTensor
fn conv2d_kernel(
    output: OutputTensor,
    input: InputTensor,
    kernel: KernelTensor,
):
    # Calculate global thread indices (output coordinates)
    var output_col = block_dim.x * block_idx.x + thread_idx.x;
    var output_row = block_dim.y * block_idx.y + thread_idx.y;

    # Calculate local thread indices
    var local_col = thread_idx.x;
    var local_row = thread_idx.y;

    # Allocate shared memory using LayoutTensorBuild
    alias shared_size = TILE_SIZE + 2 * PADDING;
    var shared_kernel = tb[dtype]().row_major[CONV_SIZE, CONV_SIZE]().shared().alloc();
    var shared_input = tb[dtype]().row_major[shared_size, shared_size]().shared().alloc();

    # Load kernel into shared memory (once per block)
    if local_row < CONV_SIZE and local_col < CONV_SIZE:
        shared_kernel[local_row, local_col] = kernel[local_row, local_col];
    
    # Calculate input coordinates (main tile loading)
    var input_row = block_idx.y * TILE_SIZE + local_row;
    var input_col = block_idx.x * TILE_SIZE + local_col;

    # Calculate shared memory position
    var shared_row = local_row + PADDING;   # Offset by padding
    var shared_col = local_col + PADDING;   # Offset by padding

    # Load main tile into shared memory
    if input_row >= 0 and input_row < INPUT_HEIGHT and input_col >= 0 and input_col < INPUT_WIDTH:
        shared_input[shared_row, shared_col] = input[input_row, input_col];
    else:
        shared_input[shared_row, shared_col] = Float32(0.0);    # Zero padding
    
    # Load boundary elements (halo regions)
    # Top boundary
    if local_row < PADDING:
        var boundary_input_row = input_row - PADDING;
        if boundary_input_row >= 0 and boundary_input_row < INPUT_HEIGHT and input_col >= 0 and input_col < INPUT_WIDTH:
            shared_input[local_row, shared_col] = input[boundary_input_row, input_col];
        else:
            shared_input[local_row, shared_col] = Float32(0.0);
    
    # Bottom boundary
    if local_row >= TILE_SIZE - PADDING:
        var boundary_input_row = input_row + PADDING;
        if boundary_input_row >= 0 and boundary_input_row < INPUT_HEIGHT and input_col >= 0 and input_col < INPUT_WIDTH:
            shared_input[shared_row + PADDING, shared_col] = input[boundary_input_row, input_col];
        else:
            shared_input[shared_row + PADDING, shared_col] = Float32(0.0);
    
    # Left boundary
    if local_col < PADDING:
        var boundary_input_col = input_col - PADDING;
        if input_row >= 0 and input_row < INPUT_HEIGHT and boundary_input_col >= 0 and boundary_input_col < INPUT_WIDTH:
            shared_input[shared_row, local_col] = input[input_row, boundary_input_col];
        else:
            shared_input[shared_row, local_col] = Float32(0.0);
    