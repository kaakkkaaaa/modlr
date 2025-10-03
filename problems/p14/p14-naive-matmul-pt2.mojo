from gpu import thread_idx, block_idx, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from sys import sizeof
from math import iota
from memory import stack_allocation, UnsafePointer
from testing import assert_equal

# Matrix dimensions (using smaller size for testing)
alias M = 32;   # Rows in matrix A and C
alias K = 32;   # Columns in A, rows in B
alias N = 32;   # Column in matrix B and C

# Block dimensions (16x16 threads per block)
alias BLOCK_SIZE_X = 16;
alias BLOCK_SIZE_Y = 16;
alias dtype = DType.float32;

fn format_float(val: Float32, precision: Int = 4) -> String:
    """Format a float to string with specified precision."""
    # For now, using String()
    return String(val)

# Matrix struct to hold data and dimensions
struct Matrix:
    var data: HostBuffer[dtype];
    var rows: Int;
    var cols: Int;

    fn __init__(out self, rows: Int, cols: Int, ctx: DeviceContext) raises:
        self.rows = rows;
        self.cols = cols;
        self.data = ctx.enqueue_create_host_buffer[dtype](
            rows * cols);
        # Need to synchronize to ensure buffer is created before use
        ctx.synchronize();
    
    fn __init__(out self, data: HostBuffer[dtype], rows: Int, cols: Int):
        self.rows = rows;
        self.cols = cols;
        self.data = data;
    
    # Improved to_string function to match PyTorch format
    fn to_string(self, max_rows: Int = 5, max_cols: Int = 5) raises -> String:
        var s = String("tensor([")
        # Handle empty matrix
        if self.rows == 0 or self.cols == 0:
            s += "])"
            return s
        
        # Determine how many rows to show before ellipsis
        var rows_before_ellipsis = min(max_rows - 1, self.rows)
        
        # Process rows before ellipsis
        for row in range(rows_before_ellipsis):
            if row > 0:
                s += ",\n "
            s += "["
            
            # Process columns for this row
            if self.cols <= max_cols:
                # Show all columns
                for col in range(self.cols):
                    if col > 0:
                        s += ", "
                    var val = self.data[row * self.cols + col]
                    s += format_float(val)  # Now calls the standalone function
            else:
                # Show first max_cols-1 elements, ellipsis, and last element
                for col in range(max_cols - 1):
                    if col > 0:
                        s += ", "
                    var val = self.data[row * self.cols + col]
                    s += format_float(val)  # Now calls the standalone function
                s += ", ..."
                # Add last element
                var last_val = self.data[row * self.cols + self.cols - 1]
                s += String(", ") + format_float(last_val)  # Now calls the standalone function
            s += "]"
        
        # Add ellipsis row if there are more rows
        if self.rows > max_rows - 1:
            s += ",\n ..."
            # Add last row
            s += ",\n ["
            
            # Process columns for last row
            if self.cols <= max_cols:
                # Show all columns
                for col in range(self.cols):
                    if col > 0:
                        s += ", "
                    var val = self.data[(self.rows - 1) * self.cols + col]
                    s += format_float(val)  # Now calls the standalone function
            else:
                # Show first max_cols-1 elements, ellipsis, and last element
                for col in range(max_cols - 1):
                    if col > 0:
                        s += ", "
                    var val = self.data[(self.rows - 1) * self.cols + col]
                    s += format_float(val)  # Now calls the standalone function
                s += ", ..."
                # Add last element
                var last_val = self.data[(self.rows - 1) * self.cols + self.cols - 1]
                s += String(", ") + format_float(last_val)  # Now calls the standalone function
            s += "]"
        
        s += "])"
        return s

fn matmul_kernel(
    A: UnsafePointer[SIMD[dtype, 1]],
    B: UnsafePointer[SIMD[dtype, 1]],
    C: UnsafePointer[SIMD[dtype, 1]],
    M: Int,
    N: Int,
    K: Int,
):
    # Calculate global thread position
    var row = block_idx.y * BLOCK_SIZE_Y + thread_idx.y;
    var col = block_idx.x * BLOCK_SIZE_X + thread_idx.x;

    # Boundary check
    if row < M and col < N:
        var sum = SIMD[dtype, 1](0.0);

        # Compute dot product of row of A and column of B
        for k in range(K):
            sum += A[row * K + k] * B[k * N + col];
        
        barrier();

        # Write result to global memory
        C[row * N + col] = sum;

# CPU matrix multiplication for verification
fn cpu_matmul(
    A: HostBuffer[dtype],
    B: HostBuffer[dtype],
    C: HostBuffer[dtype],
    M: Int,
    N: Int,
    K: Int
):
    for i in range(M):
        for j in range(N):
            var sum = SIMD[dtype, 1](0.0);
            for k in range(K):
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;


fn main() raises:
    with DeviceContext() as ctx:
        var A_host = ctx.enqueue_create_host_buffer[dtype](
            M * K);
        var B_host = ctx.enqueue_create_host_buffer[dtype](
            K * N);
        var C_host = ctx.enqueue_create_host_buffer[dtype](
            M * N);
        var C_cpu = ctx.enqueue_create_host_buffer[dtype](
            M * N);
        
        # Synchronize to ensure buffers are created
        ctx.synchronize();

        # Fill matrices with sample data
        for i in range(M * K):
            A_host[i] = Float32(i);
        
        for i in range(K * N):
            B_host[i] = Float32(i);

        for i in range(M * N):
            C_host[i] = Float32(0.0);
            C_cpu[i] = Float32(0.0);
        
        # Create matrix objects for input matrices
        var A_matx = Matrix(A_host, M, K);
        var B_matx = Matrix(B_host, K, N);

        # Print input matrices
        print("Matrix A:");
        print(A_matx.to_string());

        print("Matrix B:");
        print(B_matx.to_string());

        # Create device buffers
        var A_device = ctx.enqueue_create_buffer[dtype](M * K);
        var B_device = ctx.enqueue_create_buffer[dtype](K * N);
        var C_device = ctx.enqueue_create_buffer[dtype](M * N);

        # Copy data to device
        ctx.enqueue_copy(A_device, A_host);
        ctx.enqueue_copy(B_device, B_host);
        ctx.enqueue_memset(C_device, Float32(0.0));     # Initialize C with zeros

        # Calculate grid dimensions
        var grid_x = (N + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X;
        var grid_y = (M + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y;

        # Launch GPU kernel
        ctx.enqueue_function[matmul_kernel](
            A_device.unsafe_ptr(),
            B_device.unsafe_ptr(),
            C_device.unsafe_ptr(),
            M, N, K,
            grid_dim=(grid_x, grid_y, 1),
            block_dim=(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)
        );

        # Copy result back to host
        ctx.enqueue_copy(C_host, C_device);

        # Synchronize to ensure all operations complete
        ctx.synchronize();

        # Compute CPU version for verification
        cpu_matmul(A_host, B_host, C_cpu, M, N, K);

        # Create matrix objects for results
        var C_gpu_matrix = Matrix(C_host, M, N);
        var C_cpu_matrix = Matrix(C_cpu, M, N);

        print("\nGPU Result:");
        print(C_gpu_matrix.to_string());

        print("\nCPU Result:");
        print(C_cpu_matrix.to_string());

        # Verify results match
        var tolerance = Float32(1e-5);
        var matches = True;
        for i in range(M * N):
            if abs(C_host[i] - C_cpu[i]) > tolerance:
                matches = False;
                break
        
        if matches:
            print("\n✅ GPU and CPU results match!");
        else:
            print("\n❌ GPU and CPU results differ!");
