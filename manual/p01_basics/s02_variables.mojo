comptime MAX_USERS = 100;

def lexical_scopes():
    var num = 1;
    var dig = 1;
    if num == 1:
        print("num: ", num);
        var num = 2;
        print("num: ", num);
        dig = 2;
    print("num:", num);
    print("dig:", dig);

def main():
    var a = 5;
    var b: Float64 = 3.14;
    var c: String;
    c = "Late Init";
    print(a, b, c);

    a = 6;
    print(a);

    var parity: String;
    if a % 2 == 0:
        parity = "even";
    else:
        parity = "odd";
    print(parity);

    print("max users: ", MAX_USERS);
    lexical_scopes();