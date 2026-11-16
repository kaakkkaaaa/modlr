from std.math import isclose
from std.utils.numerics import isinf, isnan

def main():
    # Literals materialize to Int / Float64 by default
    var i = 5;
    var f = 3.3;
    var small: Int8 = 4;
    var single: Float32 = 7.5;
    print(i, f, small, single);

    var x: Int16 = 32767;
    x += 1;
    print(t"x: {x}");

    # Literal format
    print(0xFF, 0o77, 0b0111, 1_000_000, 1.2e9);

    # Fixed-width integers wrap on overflow
    var si: Int8 = 127;
    si += 1;
    var ui: UInt8 = 255;
    ui += 1;
    print("Int8 127+1 =", si, " UInt8 255+1 =", ui);

    # No implicit widening: convert explicitly.
    var my_int: Int16 = 12;
    var my_float: Float32 = 0.75;
    var result = Float32(my_int) * my_float;
    print("Float32(my_int) * my_float =", result);
    var price = 9.99;
    print("Int(price) =", Int(price), " Int(my_int) =", Int(my_int));

    # Compile-time literal math is exact; runtime Float64 math is not.
    var exact = 3.0 * (4.0 / 3.0 - 1.0);
    var three = 3.0;
    var runtime = three * (4.0 / three - 1.0);
    print(exact, runtime);

    # Floating-point comparison
    var total: Float64 = 0.0;
    for _ in range(10):
        total += 0.1;
    print(total == 1.0, isclose(total, 1.0));

    var count: Int8 = 10;
    var avg = total / Float64(count)
    print(t"avg: {avg}");

    # Special values
    var inf = FloatLiteral.infinity;
    var nan = FloatLiteral.nan;
    print(isinf(inf), isnan(nan), nan == nan);
