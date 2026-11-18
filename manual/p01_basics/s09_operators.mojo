def always_true() -> Bool:
    print("called");
    return True;

def main():
    # Arithmetic and exponentiation
    print(7 + 3, 7 - 3, 7 * 3, 2 ** 8, 2 ** 3 ** 2);

    # Division: the operand TYPES decide the rule
    var a = -7;
    var b = 4;
    print(a / b, a // b, a % b);    # Int: truncate, floor, remainder
    print(7 / 2);                   # two literals: exact compile-time math
    var x = 7.0;
    print(x / 2, x // 2);           # Float64: true division, floored float
    print(7 % 3, -7 % 4, 7 % -4);

    # Comparison, chained
    var n = 5;
    print(1 < n < 10, 1 < n < 3, 1 < n <= 5 < 9);

    # Bitwise
    var flags: UInt8 = 0b0000_0101;
    var mask: UInt8 = 0b0000_0011;
    print(flags & mask, flags | mask, flags ^ mask, ~mask);
    print(1 << 4, 16 >> 2, -16 >> 2);

    # Set, test, and clear bit 3
    var bits: UInt8 = 0b0000_0101;
    var bit3: UInt8 = 1 << 3;                # 0b0000_1000
    bits |= bit3;                            # set:   OR turns the bit on
    print(bin(bits), (bits & bit3) != 0);    # test:  AND keeps only bit 3
    bits &= ~bit3;                           # clear: AND with every bit but 3
    print(bin(bits), (bits & bit3) != 0);

    # Boolean logic uses words, and short-circuits
    var yes = True;
    var no = False;
    print(yes and no, yes or no, not yes);
    print(no and always_true());    # "called" never prints
    print(yes or always_true());    # "called" never prints

    # Membership and identity
    var colors: List[String] = ["red", "green", "blue"];
    print("red" in colors, "yellow" not in colors, "nut" in "peanut");
    var opt: Optional[Int] = None;
    print(opt is None);

    # Strings
    print("Hello" + " " + "Mojo", "ha" * 3, "Zebra" < "ant");

    # Conditional expression (condition in the middle)
    var score = 80;
    print("pass" if score > 65 else "fail");

    # Compound assignment
    var count = 0;
    count += 1;
    count *= 10;
    flags |= 0b1000_0000;
    print(count, flags);

    # Walrus: assign inside an expression
    var nums: List[Int] = [3, 1, 4];
    var size: Int;
    if (size := len(nums)) > 2:
        print("long list of", size);