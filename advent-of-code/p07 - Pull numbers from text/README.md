# p07 - Pull numbers from text

Step-by-step notes for [Pull numbers from text](https://advent.mojolang.org/pull_numbers_from_text.html). The program in this folder is `pull_numbers_from_text.mojo`.

It reads the sensor log, scans the text for runs of digits, keeps every third number as the day, checks that those days are `1` through `10`, and then parses the first temperature as a signed decimal.

## Run

`src/downloads/temp_log.txt` is a path relative to this folder.

```bash
cd "p07 - Pull numbers from text"
mojo pull_numbers_from_text.mojo
```

Expected output:

```text
  Day 1: -20.5C, Partly Cloudy  
Day 2: -22.3C, Snow
   Day 3: -19.8C, Clear
Day 4: -25.1C, Partly Cloudy
Day 5: -44.5C, BLIZZARD
Day 6: -27.7C,   Foggy
Day 7: -42.3C, partly cloudy
Day 8: -28.2C, Clear
  Day 9: -23.0C, Snow
Day 10: -36.5C, Partly Cloudy

found: [1, 20, 5, 2, 22, 3, 3, 19, 8, 4, 25, 1, 5, 44, 5, 6, 27, 7, 7, 42, 3, 8, 28, 2, 9, 23, 0, 10, 36, 5]
day numbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
complete, in order: True
scanned: [1, 20, 5]
reading: -20.5
```

## 1. Scan for digit runs

`extract_ints` walks the text one byte at a time. Digits extend the current run. Any other byte ends the run, and `Int(cur)` turns that run into a number. Adjacent digits stay together, so `20` is one value. A run at the very end is appended after the loop. `Int(text)` can fail, so the function is marked `raises`. `return nums^` transfers the list to the caller.

`text[byte=i]` is required. `text[i]` does not compile. ASCII digits are the bytes for `"0"` through `"9"`.

```mojo
from std.pathlib import Path;

def extract_ints(text: String) raises -> List[Int]:
    var nums = List[Int]();
    var cur = String("");
    for i in range(text.byte_length()):
        var c = String(text[byte=i]);
        if c >= "0" and c <= "9":
            cur += c;
        elif cur:
            nums.append(Int(cur));
            cur = String("");
    if cur:
        nums.append(Int(cur));
    return nums^;
```

Read the log and scan the whole file. The minus signs and decimal points are not digits, so `-20.5` becomes `20` and `5`.

```mojo
def main() raises:
    var log = Path("src/downloads/temp_log.txt");
    var text = log.read_text();
    var lines = text.splitlines();
    print(text);

    var found = extract_ints(text);
    print(t"found: {found}");
```

## 2. Keep the day numbers

Each line contributes three numbers: the day, the whole degrees, and the tenths digit. `range(0, len(found), 3)` visits every third index, which is the day.

```mojo
    var day_numbers = List[Int]();
    for i in range(0, len(found), 3):
        day_numbers.append(found[i]);
    print(t"day numbers: {day_numbers}");
```

## 3. Check the days

Ten lines should produce the days `1` through `10`, in order. Build that sequence and compare.

```mojo
    var expected = [i for i in range(1, len(lines) + 1)];
    print(t"complete, in order: {day_numbers == expected}");
```

## 4. Parse one signed decimal

A digit scan of the first line returns `[1, 20, 5]`. The sign is dropped and the dot splits `-20.5` into two numbers. Split the line into fields instead, remove the trailing `C`, and parse the field with `Float64`.

```mojo
    print(t"scanned: {extract_ints(String(lines[0]))}");

    var entry = lines[0].strip();
    var temp_field = entry.split(": ")[1].split(", ")[0];
    var reading = Float64(temp_field.removesuffix("C"));
    print(t"reading: {reading}");
```

`removesuffix("C")` removes the unit only when it is at the end. The temperature field in the log ends with `C`.

## Topics covered

Digit runs, `text[byte=i]`, `Int` and `Float64` parsing, `raises`, returning a list with `^`, taking every third value, and checking the days against `1` through `10`.
