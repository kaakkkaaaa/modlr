# p03 - Mojo basics

Step-by-step reconstruction of `analyzer.mojo`, following [Mojo basics](https://advent.mojolang.org/mojo_basics.html).

The finished program reads four North Pole temperatures, prints each day, averages them, and labels the week. An empty list raises an error instead of dividing by zero.

## Run

From `advent-of-code`:

```bash
mojo "p03 - Mojo basics/analyzer.mojo"
```

From this folder:

```bash
cd "p03 - Mojo basics"
mojo analyzer.mojo
```

Expected output:

```text
North Pole Temperature Analyzer
Recorded 4 temperatures
 Day 1: -20.5°C
 Day 2: -22.3°C
 Day 3: -19.8°C
 Day 4: -25.1°C
Average temperature: -21.92°C
Status: Comfortable week
```

## 1. Hello Mojo

Create `analyzer.mojo`. Every Mojo program starts at `main()`.

```mojo
def main():
    print("North Pole Temperature Analyzer");
```

Run `mojo analyzer.mojo`. If that line prints, the toolchain is working.

## 2. Variables and data

Store the four sensor readings in a `List[Float64]`. The square brackets fix the element type at compile time. A `t"..."` string is a template: Mojo fills each `{...}` brace with the expression inside it.

```mojo
def main():
    print("North Pole Temperature Analyzer");

    # Square brackets tell Mojo the `List` type at compile time.
    var temps: List[Float64] = [-20.5, -22.3, -19.8, -25.1];
    print(t"Recorded {len(temps)} temperatures");
```

## 3. Loops

`range(len(temps))` yields the indexes `0, 1, 2, 3`. Add one so the printed day numbers start at 1.

```mojo
    for index in range(len(temps)):
        print(t" Day {index + 1}: {temps[index]}°C");
```

`for` also supports `break`, `continue`, and `return`. Mojo has `while` loops as well.

## 4. Functions

Put the average in its own function so later days can reuse it. The return type goes after `->`. Divide by `Float64(count)` so the result stays a `Float64`.

```mojo
def calculate_average(temps: List[Float64]) -> Float64:
    var total: Float64 = 0.0;
    var count: Int = 0;
    for temp in temps:
        total += temp;
        count += 1;
    return total / Float64(count);
```

Call it at the end of `main()`:

```mojo
    var avg = calculate_average(temps);
    print(t"Average temperature: {round(avg, 2)}°C"); # Average: -21.92
```

## 5. Conditionals

Classify the week from that average. Mojo checks the branches in order, so the `-20.0` test has to come before the `-25.0` test.

```mojo
    if avg > -20.0:
        print("Status: Hot week");
    elif avg > -25.0:
        print("Status: Comfortable week");
    else:
        print("Status: Cool week");
```

With these readings the average is about `-21.92`, so the program prints `Status: Comfortable week`.

## 6. Raise errors

An empty list would divide by zero. Mark the function with `raises` and stop before the loop when there is no data.

```mojo
def calculate_average(temps: List[Float64]) raises -> Float64:
    if len(temps) == 0:
        raise Error("No temperatures data");

    var total: Float64 = 0.0;
    var count: Int = 0;
    for temp in temps:
        total += temp;
        count += 1;
    return total / Float64(count);
```

## 7. Handle errors

Wrap the average and the status check in `try` / `except` so a missing report prints a message instead of crashing.

```mojo
    try:
        var avg = calculate_average(temps);
        print(t"Average temperature: {round(avg, 2)}°C"); # Average: -21.92

        if avg > -20.0:
            print("Status: Hot week");
        elif avg > -25.0:
            print("Status: Comfortable week");
        else:
            print("Status: Cool week");
    except e:
        print(t"Error: {e}");
```

To see the error path, replace the temperature list with an empty one:

```mojo
    var temps: List[Float64] = [];
```

The program should print `Error: No temperatures data`. Put the four readings back when you are done. That empty assignment is left in `analyzer.mojo` as a comment.

## 8. Final code

`analyzer.mojo` in this folder is the program those steps produce.

## Topics covered

Variables, lists, `for` loops, functions, conditionals, and `raises` / `try` / `except`.
