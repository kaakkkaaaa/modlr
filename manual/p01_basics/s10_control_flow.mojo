def main():
    # if / elif / else
    var temp_celcius = 25;
    if temp_celcius <= 0:
        print("It's freezing!");
    elif temp_celcius < 20:
        print("It's cool.");
    elif temp_celcius < 30:
        print("It's warm.");
    else:
        print("It's hot!");
    if temp_celcius > 20: print("One-line if works too.");

    # while, with tuple assignment
    var fib_prev = 0;
    var fib_curr = 1;
    print(fib_prev, end="");
    while fib_curr < 50:
        print(",", fib_curr, end="");
        fib_prev, fib_curr = fib_curr, fib_prev + fib_curr;
    print();

    # continue / break
    var n = 0;
    while n < 5:
        n += 1;
        if n == 3:
            continue;
        print(n, end=", ");
    print();

    # while ... else: else runs when the condition becomes False
    var k = 5;
    while k < 4:
        k += 1;
    else:
        print("while finished without break");
    
    # for over a range and over collections
    for i in range(0, 10, 3):
        print(i, end=" ");
    print();
    for i in range(3, 0, -1):
        print(i, end=" ");
    print();

    var states: List[String] = ["California", "Hawaii", "Oregon"];
    for state in states:
        print(state, end=" ");
    print();
    for i, state in enumerate(states):
        print(i, state, end="; ");
    print();

    # for ... else: else is skipped on break
    var animals: List[String] = ["cat", "aardvark", "dog"];
    for animal in animals:
        if animal == "dog":
            print("Found a dog");
            break;
    else:
        print("No dog found");
    
    # Mutating elements needs `ref`
    var values: List[Int] = [1, 4, 7, 3, 6, 11];
    for ref value in values:
        if value % 2 != 0:
            value -= 1;
    print(values);

    # Exercise: primes below 50 with nested for, break, and for ... else
    for num in range(2, 50):
        for d in range(2, num):
            if num % d == 0:
                break;              # found a divisor: not prime, skip the else
        else:
            print(num, end=" ");    # loop ran out without a break: prime
    print();

    # Exercise: Dict items() filtered by a threshold
    var scores: Dict[String, Int] = {"ada": 92, "bob": 61, "cy": 78, "dee": 45};
    var threshold = 70;
    for item in scores.items():
        if item.value > threshold:
            print(item.key, item.value);
