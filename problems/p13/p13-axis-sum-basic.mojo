from layout import LayoutTensor, Layout
from layout.math import sum

fn main() raises:
    # Create a simple 2x3 tensor with sample data
    data = InlineArray[Int32, 6](0, 1, 2, 3, 4, 5);
    tensor = LayoutTensor[DType.int32, Layout.row_major(2, 3)](data);

    # Display the original tensor (will show as 2x3 matrix)
    print("Original Tensor:");
    print(tensor)

    # Sum along axis 0 (collapse rows, keep columns)
    # This gives us [3, 5, 7] because:
    # - Column 0: 0 + 3 = 3
    # - Column 1: 1 + 4 = 5  
    # - Column 2: 2 + 5 = 7
    result_axis0 = sum[0](tensor);
    print("Sum along axis 0 (collapse rows):");
    print(result_axis0);

    # Sum along axis 1 (collapse columns, keep rows)
    # This would give us [3, 12] because:
    # - Row 0: 0 + 1 + 2 = 3
    # - Row 1: 3 + 4 + 5 = 12
    result_axis1 = sum[1](tensor);
    print("Sum along axis 1 (collapse columns):");
    print(result_axis1);