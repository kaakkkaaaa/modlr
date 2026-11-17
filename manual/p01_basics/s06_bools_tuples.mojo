def min_max(a: Int, b: Int) -> Tuple[Int, Int]:
    return (a, b) if a < b else (b, a)

def divmod_int(a: Int, b: Int) -> Tuple[Int, Int]:
    return (a / b, a % b);

def main():
    # Booleans
    var condition_a = False;
    var condition_b: Bool;
    condition_b = not condition_a;
    print(condition_a, condition_b);

    # Truthiness: zero, empty strings, empty collections are False
    var empty: List[Int] = [];
    var name = "Mojo";
    print(Bool(0), Bool(7), Bool(""), Bool(name), Bool(empty));
    if name:
        print("name is set");
    
    # Tuples: fixed size, mixed types
    var pair = Tuple[Int, String](1, "Example");
    var x, y = pair;
    print(x, y);
    print(pair[1]);

    var div, mod = divmod_int(10, 3);
    print("div: ", div, "mod: ", mod);

    var point = (3, 4.5, "label");
    print(point[0], point[1], point[2]);
    point[0] = 30;
    print(point[0]);

    var lo, hi = min_max(9, 2);
    print("lo: ", lo, "hi: ", hi);

    # Swap with tuple assignment
    var a = 1;
    var b = 2;
    a, b = b, a;
    print(a, b);