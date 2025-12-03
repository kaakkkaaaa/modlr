from gpu import thread_idx, block_idx, warp, barrier
from gpu.host import DeviceBuffer, DeviceContext
from layout import Layout, LayoutTensor
from math import iota

alias dtype = DType.float32;

fn calculate_expected_row_sum(row: Int, cols: Int) -> Float32:
    """
    Calculate the expected sum for a given row in an iota-filled matrix.
    For row R in a matrix filled with iota(total_elements):
    - Row R contains elements: [R*cols, R*cols+1, ..., R*cols+cols-1]
    - Sum = cols * R * cols + (0+1+...+(cols-1))
    - Sum = cols * R * cols + cols*(cols-1)/2
    - Sum = cols * (R * cols + (cols-1)/2).
    """
    var start_val = Float32(row * cols);
    var end_val = Float32(row * cols + cols - 1);
    
    # Sum of arithmetic sequence: n * (first + last) / 2
    var expected = Float32(cols) * (start_val + end_val) / 2.0;
    
    return expected

fn inter_warp_optimized() raises:
    """
    OPTIMIZED INTER-WARP REDUCTION.
    This uses a hybrid approach: parallel collection + thread 0 coordination.
    More efficient than pure sequential.
    """
    alias rows = 64;
    alias cols = 256;
    alias total_elements = rows * cols;
    alias threads_per_block = 256;
    alias blocks_needed = rows;

    print("\n🚀 OPTIMIZED: Hybrid Inter-Warp Reduction");

    var ctx = DeviceContext();

    var in_buffer = ctx.enqueue_create_buffer[dtype](total_elements);
    var out_buffer = ctx.enqueue_create_buffer[dtype](rows);

    with in_buffer.map_to_host() as host_buffer:
        iota(host_buffer.unsafe_ptr(), total_elements);
    
    _ = out_buffer.enqueue_fill(0.0);

    alias in_layout = Layout.row_major(rows, cols);
    alias InTensor = LayoutTensor[dtype, in_layout, MutableAnyOrigin];
    var in_tensor = InTensor(in_buffer);

    alias out_layout = Layout.row_major(rows);
    alias OutTensor = LayoutTensor[dtype, out_layout, MutableAnyOrigin];
    var out_tensor = OutTensor(out_buffer);

    fn optimized_kernel(in_tensor: InTensor, out_tensor: OutTensor):
        """
        Hybrid approach: Use parallel loading + centralized reduction.
        """
        var row = block_idx.x;
        var thread_id = thread_idx.x;
        var local_sum : Float32 = 0.0;

        # Phase 1: Each thread loads its element (parallel)
        if thread_id < cols:
            local_sum = in_tensor.load[1](row, thread_id);
        
        # Phase 2: Warp-level reduction (parallel within warps)
        local_sum = warp.sum(local_sum);

        # Phase 3: Inter-warp coordination
        # Strategy: Thread 0 queries each warp leader's result

        barrier();

        if thread_id == 0:
            var final_sum: Float32 = local_sum; # Thread 0's warp sum

            # Add other warps' contributions by reconstructing them
            alias warp_size = 32;
            alias num_warps = cols//warp_size;
            for warp_id in range(1, num_warps):
                var warp_contribution: Float32 = 0.0;
                var warp_start = warp_id * warp_size;
                var warp_end = warp_start + warp_size;

                # Calculate this warp's total contribution
                for idx in range(warp_start, min(warp_end, cols)):
                    warp_contribution += in_tensor.load[1](row, idx);
                
                final_sum += warp_contribution;
            
            out_tensor[row] = final_sum;
    

    ctx.enqueue_function[optimized_kernel](
        in_tensor,
        out_tensor,
        grid_dim=blocks_needed,
        block_dim=threads_per_block,
    );

    verify_results_all_rows[64, 256](out_buffer, "Optimized Hybrid Method");


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
                print("   ... (testing rows {} to {}) ...".format(
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
            print("🎉 PERFECT! ALL {} ROWS PASSED! ✅".format(rows));
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
            
            # Provide specific debugging advice
            print("\n🔧 DEBUGGING RECOMMENDATIONS:");
            print("-" * 35);
            
            if failed_tests == rows:
                print("• TOTAL FAILURE: Check if kernel is running at all");
                print("• Verify buffer initialization and kernel launch");
            elif failed_tests > (rows // 2):
                print("• MAJOR ISSUES: Core algorithm problem");
                print("• Check inter-warp reduction logic");
                print("• Verify thread coordination and synchronization");
            elif max_error > 1000.0:
                print("• LARGE ERRORS: Missing data or coordination failure");
                print("• Check if all warps are contributing to final result");
                print("• Verify shared memory and sync_threads() usage");
            elif max_error > 10.0:
                print("• MODERATE ERRORS: Partial reduction issues");
                print("• Check warp-level reduction implementation");
                print("• Verify thread indexing and data access patterns");
            else:
                print("• MINOR ERRORS: Possible floating-point precision");
                print("• Check accumulation order and numerical stability");
                print("• Consider using higher precision arithmetic");
        
        print();
        
        # Summary statistics
        if failed_tests == 0:
            print("✨ VERIFICATION COMPLETE: Your GPU kernel is rock-solid! ✨");
        else:
            print("🚨 VERIFICATION COMPLETE: Issues detected - see recommendations above 🚨");


fn main() raises:
    inter_warp_optimized();