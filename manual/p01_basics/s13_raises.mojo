def validate(value: Int) raises:
    if value < 0:
        raise Error("Value must be non-negative");

def checked_sqrt_floor(value: Int) raises -> Int:
    validate(value);
    var r = 0;
    while (r + 1) * (r + 1) <= value:
        r += 1;
    return r;

def safe_report(value: Int):
    try:
        print("floor sqrt of", value, "=", checked_sqrt_floor(value));
    except e:
        print("Handled:", e);
    else:
        print("  (no error)");
    finally:
        print("  (always runs)");

def parse_age(s: String) raises -> Int:
    var age = Int(s);
    if age < 0:
        raise Error(t"Age can't be negative: {age}");
    return age;

def show_age(s: String):
    try:
        var age = parse_age(s);
        print(t"Age: {age}");
    except e:
        print(t"Sorry, '{s}' isn't a valid age ({e})");

def main() raises:
    safe_report(17);
    safe_report(-4);
    print(checked_sqrt_floor(81));
    _ = checked_sqrt_floor(-1);
    print("never printed");

    show_age("42");
    show_age("-3");
    show_age("abc");
