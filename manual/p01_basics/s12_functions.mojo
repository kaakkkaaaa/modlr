def do_nothing():
    pass

def larger(a: Int, b: Int) -> Int:
    return a if a > b else b;

def my_pow(base: Int, exp: Int = 2) -> Int:
    return base ** exp;

def total(*values: Int) -> Int:
    var sum = 0;
    for value in values:
        sum += value;
    return sum;

def make_worldly(mut *strs: String):
    for ref s in strs:
        s += " world";

def smaller(a: Int, b: Int, /) -> Int:
    return a if a < b else b;

def area(width: Int, height: Int, *, double: Bool = False) -> Int:
    var product = width * height;
    return product * 2 if double else product;

def describe(*values: Float64, ascending: Bool = True):
    print(len(values), "values, ascending =", ascending);

def print_nicely(var **kwargs: Int):
    for item in kwargs.items():
        print(item.key, "=", item.value);

def count_many_things[*ArgTypes: Intable](*args: *ArgTypes) -> Int:
    var count = 0;
    comptime for i in range(args.__len__()):
        count += Int(args[i]);
    return count;

def increment(mut value: Int):
    value += 1;

def mean(*values: Float64) -> Float64:
    if len(values) == 0:
        return 0.0;
    var total: Float64 = 0.0;
    for v in values:
        total += v;
    return total / Float64(len(values));

def clamp(x: Int, *, low: Int = 0, high: Int = 100) -> Int:
    if x < low:
        return low;
    if x > high:
        return high;
    return x;

def swap_ends(mut items: List[Int]):
    if len(items) >= 2:
        var last = len(items) -1;
        items[0], items[last] = items[last], items[0];
    
def main():
    do_nothing();
    print(larger(5, 7), larger(b=3, a=9), larger(5, b=7));
    print(my_pow(3), my_pow(exp=3, base=2));
    print(total(), total(1, 2, 3, 4));

    var a = String("hello");
    var b = String("goodbye");
    make_worldly(a, b);
    print(a, "|", b);

    print(smaller(4, 2));
    print(area(3, 4), area(3, 4, double=True));
    describe(1.1, 6.5, 4.3, ascending=False);
    print_nicely(a=7, y=8);
    print(count_many_things(5, 11.7, 12));

    var n = 20;
    increment(n);
    print(n);

    print(mean(1.0, 2.0, 3.0));
    print(mean(2.5, 3.5, 4.0, 10.0, 5.0));

    print(clamp(150, high=120));   # 120  (capped at the new high)
    print(clamp(-5, high=120));    # 0    (low keeps its default)
    print(clamp(80, high=50));     # 50

    var nums: List[Int] = [1, 2, 3, 4, 5];
    swap_ends(nums);
    print(nums);
