from gpu import thread_idx, block_idx, warp, barrier
from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from math import iota

alias dtype = DType.float32;

# ===== VERIFICATION FUNCTIONS =====

fn calculate_expected_row_sum(row: Int, cols: Int) -> Float32:
    """
    Calculate the expected sum for a given row in an iota-filled matrix.
    For row R in a matrix filled with iota(total_elements):
    - Row R contains elements: [R*cols, R*cols+1, ..., R*cols+cols-1]
    - Sum = cols * (first_element + last_element) / 2.
    """
    var start_val = Float32(row * cols);
    var end_val = Float32(row * cols + cols - 1);
    
    # Sum of arithmetic sequence: n * (first + last) / 2
    var expected = Float32(cols) * (start_val + end_val) / 2.0;
    
    return expected;

fn verify_results_all_rows[
    rows: Int, cols: Int
](out_buffer: DeviceBuffer[dtype], method_name: String) raises:
    """
    Verify all rows - tests every single row for complete validation.
    Optimized specifically for testing all rows without sampling logic.
    """
    with out_buffer.map_to_host() as host_buffer:
        print(String("\n🔍 {} - COMPLETE VERIFICATION (ALL {} ROWS)").format(method_name, rows));
        print("=" * 80);
        
        # Statistics tracking
        var total_errors = Float32(0.0);
        var max_error = Float32(0.0);
        var max_error_row = Int(-1);
        var passed_tests = Int(0);
        var failed_tests = Int(0);
        var tolerance = Float32(1.0);
        
        # Test every single row
        print("Testing all rows for correctness...");
        print();
        
        # Show detailed results for first few and last few rows
        var show_details_for_first = min(5, rows);  # Show first 5
        var show_details_for_last = min(5, rows);   # Show last 5
        
        for row in range(rows):
            var expected = calculate_expected_row_sum(row, cols);
            var actual = host_buffer[row];
            var error = abs(expected - actual);
            var percent_error = Float32(0.0);
            if expected != 0.0:
                percent_error = (error / expected) * 100.0;
            
            # Update statistics
            total_errors = total_errors + error;
            if error > max_error:
                max_error = error;
                max_error_row = row;
            
            # Count pass/fail
            if error < tolerance:
                passed_tests = passed_tests + 1;
            else:
                failed_tests = failed_tests + 1;
            
            # Show detailed output for first few, last few, and any failures
            var should_show_details = (
                row < show_details_for_first or 
                row >= (rows - show_details_for_last) or
                error >= tolerance
            );
            
            if should_show_details:
                var status_icon = "✅" if error < tolerance else "❌";
                var status_text = "PASS" if error < tolerance else "FAIL";
                
                print(String("{} Row {}: {} | Expected: {} | Error: {} ({}%) | {}").format(
                    status_icon, row, actual, expected, error, percent_error, status_text
                ));
            elif row == show_details_for_first:
                # Show ellipsis after first batch
                print(String("   ... (testing rows {} to {}) ...").format(
                    show_details_for_first, rows - show_details_for_last - 1
                ));
        
        print();
        print("📈 COMPLETE STATISTICAL SUMMARY:");
        print("-" * 40);
        
        var mean_error = total_errors / Float32(rows);
        var success_rate = (Float32(passed_tests) / Float32(rows)) * 100.0;
        
        print(String("• Total Rows Tested: {} (100%)").format(rows));
        print(String("• Passed: {} | Failed: {}").format(passed_tests, failed_tests));
        print(String("• Success Rate: {}%").format(success_rate));
        print(String("• Mean Error: {}").format(mean_error));
        print(String("• Max Error: {} (Row {})").format(max_error, max_error_row));
        print(String("• Tolerance: {}").format(tolerance));
        
        print();
        print("🎯 FINAL VERDICT:");
        print("-" * 20);
        
        if failed_tests == 0:
            print(String("🎉 PERFECT! ALL {} ROWS PASSED! ✅").format(rows));
            print(String("   {} method is working flawlessly across ALL test cases!").format(method_name));
            if mean_error < 0.001:
                print("   🏆 OUTSTANDING: Machine precision accuracy!");
            elif mean_error < 0.01:
                print("   🌟 EXCELLENT: Extremely low numerical error!");
            elif mean_error < 1.0:
                print("   👍 GOOD: Acceptable numerical precision!");
        else:
            print(String("⚠️  {} OUT OF {} ROWS FAILED! ❌").format(failed_tests, rows));
            print(String("   {} method has issues ({:.1f}% failure rate)").format(
                method_name, (Float32(failed_tests) / Float32(rows)) * 100.0
            ));
            print(String("   Worst error: {} in row {}").format(max_error, max_error_row));
        
        print();
        
        # Summary statistics
        if failed_tests == 0:
            print("✨ VERIFICATION COMPLETE: Your GPU kernel is rock-solid! ✨");
        else:
            print("🚨 VERIFICATION COMPLETE: Issues detected - see recommendations above 🚨");


fn scalable_axis_sum() raises:
    """
    Demonstrates axis sum for larger tensors, showing how to handle
    cases where tensor dimensions exceed GPU hardware limits.
    
    This example processes a 64x256 matrix, which is more representative
    of real machine learning workloads.
    """

    alias rows = 64;
    alias cols = 256;
    alias total_elements = rows * cols;

    # For larger tensors, we need to think about GPU resource limits
    # Most GPUs have limits on threads per block (often 1024)
    alias threads_per_block = 32;   # Use 32 threads per block
    alias blocks_needed = rows;     # One block per row for axis=1 reduction

    print(String("Processing {}x{} matrix for axis=1 sum").format(rows, cols));
    print(String("Using {} blocks with {} threads each").format(blocks_needed, threads_per_block));

    var ctx = DeviceContext();

    # Allocate buffers for the larger tensor
    var in_buffer = ctx.enqueue_create_buffer[dtype](total_elements);
    var out_buffer = ctx.enqueue_create_buffer[dtype](rows);

    with in_buffer.map_to_host() as host_buffer:
        iota(host_buffer.unsafe_ptr(), total_elements);

    _ = out_buffer.enqueue_fill(0);

    # Create tensor layouts
    alias in_layout = Layout.row_major(rows, cols);
    alias InTensor = LayoutTensor[dtype, in_layout, MutableAnyOrigin];
    var in_tensor = InTensor(in_buffer);

    alias out_layout = Layout.row_major(rows);
    alias OutTensor = LayoutTensor[dtype, out_layout, MutableAnyOrigin];
    var out_tensor = OutTensor(out_buffer);

    fn large_tensor_reduction_kernel(in_tensor: InTensor, out_tensor: OutTensor):
        """
        Advanced reduction kernel that handles cases where the number of
        elements to reduce exceeds the number of threads in a block.
        
        Strategy: Each thread processes multiple elements, then we perform
        a final warp-level reduction to combine results.
        """

        var row = block_idx.x;
        var thread_id = thread_idx.x;
        var local_sum: Float32 = 0.0;

        # Each thread may need to process multiple elements
        # We stride through the row by the number of threads
        var start_col = thread_id;
        while start_col < cols:
            local_sum += in_tensor.load[1](row, start_col);
            start_col += threads_per_block;
        
        # Now perform warp-level reduction to combine partial sums
        local_sum = warp.sum(local_sum);
        
        # For now, only thread 0 writes (this will still give wrong result)
        # This demonstrates why we need proper inter-warp reduction
        if thread_id == 0:
            out_tensor[row] = local_sum

    
    # Launch the kernel
    ctx.enqueue_function[large_tensor_reduction_kernel](
        in_tensor,
        out_tensor,
        grid_dim=blocks_needed,
        block_dim=threads_per_block,
    );

    verify_results_all_rows[rows, cols](out_buffer, "Scalabel Axis Sum");


fn main() raises:
    scalable_axis_sum();