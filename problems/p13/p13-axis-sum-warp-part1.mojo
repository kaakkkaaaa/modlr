from gpu import thread_idx, block_idx, warp
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from math import iota

alias dtype = DType.float32;
alias BLOCKS = 8;   # Number of blocks (think of these as rows to process)
alias THREADS = 4;  # Number of threads per block (elements per row)
alias ELEMENTS_IN = BLOCKS * THREADS;

fn axis_sum_warp() raises:
    """
    Demonstrates axis sum reduction on GPU using manual kernel implementation.
    
    We'll create an 8x4 matrix and sum along axis 1 (reduce columns within each row).
    Each block handles one row, and threads within that block cooperatively 
    sum the elements in that row.
    """
    # Step 1: Create GPU context for managing device operations
    var ctx = DeviceContext();

    # Step 2: Allocate GPU memory buffers
    # Input buffers: 8x4 = 32 elements total
    var in_buffer = ctx.enqueue_create_buffer[dtype](ELEMENTS_IN);
    # Ouput buffers: 8 elements (one sum per row)
    var out_buffer = ctx.enqueue_create_buffer[dtype](BLOCKS);

    # Step 3: Initialize input data and copy to GPU
    # This creates sequential values: [0, 1, 2, 3, 4, 5, ...]
    with in_buffer.map_to_host() as host_buffer:
        iota(host_buffer.unsafe_ptr(), ELEMENTS_IN);
    
    # Step 4: Zero the output buffer (important for reduction operations)
    _ = out_buffer.enqueue_fill(0);

    # Step 5: Create tensor view for structured access
    # Input tensor : 8 rows x 4 columns
    alias layout = Layout.row_major(BLOCKS, THREADS);
    alias InTensor = LayoutTensor[dtype, layout, MutableAnyOrigin];
    var in_tensor = InTensor(in_buffer);

    # Output tensor: 8 elements (one per row)
    alias out_layout = Layout.row_major(BLOCKS);
    alias OutTensor = LayoutTensor[dtype, out_layout, MutableAnyOrigin];
    var out_tensor = OutTensor(out_buffer);

    # Step 6: Define the GPU kernel function
    fn reduce_sum_kernel(
        in_tensor: InTensor, out_tensor: OutTensor
    ):
        """
        GPU kernel that performs axis=1 sum reduction.
        
        Key concepts:
        - block_idx.x tells us which row (block) we're processing
        - thread_idx.x tells us which column (thread) within that row
        - warp.sum() performs efficient parallel reduction within a warp
        """

        # Each thread loads one element from its assigned position
        var value = in_tensor.load[1](block_idx.x, thread_idx.x);

        # Warp-level reduction: efficiently sum values across threads
        # This is much faster than sequential addition
        value = warp.sum(value);

        # Only thread 0 in each block writes the final result
        # This prevents race conditions and duplicate writes
        if thread_idx.x == 0:
            out_tensor[block_idx.x] = value;
    
    # Step 7: Launch the kernel on GPU
    ctx.enqueue_function[reduce_sum_kernel](
        in_tensor,
        out_tensor,
        grid_dim=BLOCKS,
        block_dim=THREADS,
    );

    # Step 8: Copy results back to CPU and display
    with out_buffer.map_to_host() as host_buffer:
        print("Input matrix (8x4):");
        print("Row 0: [0, 1, 2, 3] -> Sum = 6");
        print("Row 1: [4, 5, 6, 7] -> Sum = 22");
        print("Row 2: [8, 9, 10, 11] -> Sum = 38");
        print("... and so on");
        print("\nAxis=1 sum results:");
        print(host_buffer);

fn main() raises:
    axis_sum_warp();