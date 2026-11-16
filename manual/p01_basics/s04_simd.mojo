from std.utils.numerics import max_finite, min_finite

def describe[dtype: DType]():
    print(dtype, "is floating-point:", dtype.is_floating_point());
    print(dtype, "min/max:", min_finite[dtype](), max_finite[dtype]());

def main():
    var vec1 = SIMD[DType.int8, 4](2, 3, 5, 7);
    var vec2 = SIMD[DType.int8, 4](1, 2, 3, 4);
    print(vec1 * vec2);     # elementwise
    print(vec1 + 10);       # scalar broadcast
    print(vec1[2]);         # read one lane
    vec1[0] = 100;          # write one lane
    print(vec1);

    # A Float32 is just a 1-lane SIMD.
    var x: Float32 = 1.5;
    var same: SIMD[DType.float32, 1] = x;
    print(same);

    # Converting between element types
    var f = SIMD[DType.float32, 4](2.2, 3.3, 4.4, 5.5);
    var n = SIMD[DType.int16, 4](-1, 2, -3, 4);
    print(f * n.cast[DType.float32]());
    print(n + SIMD[DType.int16, 4](f));

    var f64 = SIMD[DType.float64, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    print(f64.reduce_add());

    # Reductions and per-lane comparisons
    print(f.reduce_add(), f.reduce_max());
    var mask = f.gt(3.0);
    print(mask);
    print(mask.select(f, SIMD[DType.float32, 4](0)));

    var mask_neg = n.lt(0);
    print(mask_neg);
    print(mask_neg.select(n, SIMD[DType.int16, 4](0)));

    # .float32 is shorthand for DType.float32 where a DType is expected
    var g = SIMD[.float32, 2](1.0, 2.0);
    print(g);

    describe[DType.uint8]();