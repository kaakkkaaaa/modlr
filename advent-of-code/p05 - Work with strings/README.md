# p05 - Work with strings

Step-by-step notes for [Work with strings](https://advent.mojolang.org/work_with_strings.html). You write `string_tour.mojo` yourself as you go. This folder does not include that program.

The program you end up with reads the first line of the sensor log, then measures it, walks the characters, joins them back together, trims it, reverses it, slices it, searches it, and splits it into fields.

## Run

After you have written `string_tour.mojo` here, run it from this folder with the Mojo already on your path. `src/downloads/temp_log.txt` is a path relative to the current directory.

```bash
cd "p05 - Work with strings"
mojo string_tour.mojo
```

`src/downloads/temp_log.txt` is the same sensor log as Work with files. Line 1 is 32 bytes on purpose: two spaces before `Day` and two spaces after `Cloudy`. If a later length prints `30`, those two trailing spaces were removed when the file was saved. Put them back.

Expected output:

```text
string: '  Day 1: -20.5C, Partly Cloudy  '
  Day 1: -20.5C, Partly Cloudy   length: 32
[' ', ' ', 'D', 'a', 'y', ' ', '1', ':', ' ', '-', '2', '0', '.', '5', 'C', ',', ' ', 'P', 'a', 'r', 't', 'l', 'y', ' ', 'C', 'l', 'o', 'u', 'd', 'y', ' ', ' ']
'  Day 1: -20.5C, Partly Cloudy  '
'Day1:-20.5C,PartlyCloudy'
'H, e, l, l, o'
cleaned: 'Day 1: -20.5C, Partly Cloudy', bytes: 28
'Day 1: -20.5C, Partly Cloudy'
Reversed bytes: 'yduolC yltraP ,C5.02- :1 yaD'
'yduolC yltraP ,C5.02- :1 yaD'
byte prefix:     'Day 1'
byte suffix:     'loudy'
byte substring:  'y 1: -20'
contains 'Day': True
position of ':': 5
position of '!': -1
starts with 'Day': True
ends with 'Cloudy': True
label: 'Day 1'
Not found
This is printed
Processing: 'Day 1: -20.5C, Partly Cloudy'
Day 1: -20.5C, Cloudy
Day 1: -20.5, loudy
split on ', ': [Day 1: -20.5C, Partly Cloudy]
count: 2
partly cloudy
PARTLY CLOUDY
'cloudy' matches: True
```

## 1. Read the first line

Create `string_tour.mojo`. Advent input is ASCII, so one byte is one character. Read the log and keep only line 0. The quotes in the print make the extra spaces visible. `read_text()` can fail, so `main()` is marked `raises`.

```mojo
from std.pathlib import Path;

def main() raises:
    var log = Path("src/downloads/temp_log.txt");
    var string: String = String(log.read_text().splitlines()[0]);
    print(t"string: '{string}'");
```

`splitlines()[0]` is a view into the file text. `String(...)` copies that first line so you can keep using it.

## 2. Measure it

`byte_length()` is the length for ASCII text. A template string starts with `t` and fills each `{...}` when it is printed. It is a template, so turn it into a `String` before you store it.

```mojo
    var length = string.byte_length();
    print(t"{string} length: {length}"); # 32
```

The printed line still shows the two leading spaces, then `length: 32`.

## 3. Walk the characters

`codepoint_slices()` yields one view per character. On this ASCII line, each view is one byte.

```mojo
    print([String(t"'{slice}'") for slice in string.codepoint_slices()]);
```

The list starts with the two spaces and ends with the two spaces.

## 4. Join the pieces

`join()` is called on the separator. An empty separator glues the pieces together. A separator is inserted only between elements.

```mojo
    var joined = "".join([slice for slice in string.codepoint_slices()]);
    print(t"'{joined}'"); # '  Day 1: -20.5C, Partly Cloudy  '

    joined = "".join(
        [slice for slice in string.codepoint_slices() if " " not in slice]
    );
    print(t"'{joined}'"); # 'Day1:-20.5C,PartlyCloudy'

    var hello: String = "Hello";
    joined = ", ".join([slice for slice in hello.codepoint_slices()]);
    print(t"'{joined}'"); # 'H, e, l, l, o'
```

## 5. Trim the edges

`strip()` returns a view with ASCII whitespace removed from both ends. `lstrip()` cleans the left side, and `rstrip()` cleans the right. The length goes from 32 to 28. Later steps use `cleaned`.

```mojo
    var cleaned = string.strip();
    print(t"cleaned: '{cleaned}', bytes: {cleaned.byte_length()}");
```

`strip(chars)` removes those characters instead of whitespace:

```mojo
    var flagged = "**" + cleaned + "**";
    print(t"'{flagged.strip("*")}'"); # 'Day 1: -20.5C, Partly Cloudy'
```

## 6. Reverse it

Walk the byte indexes from the end, and also walk `codepoint_slices_reversed()`. On ASCII text both produce the same string.

```mojo
    var s: String = "";

    for index in reversed(range(cleaned.byte_length())):
        s = s + String(cleaned[byte=index]);
    print(t"Reversed bytes: '{s}'"); # 'yduolC yltraP ,C5.02- :1 yaD'

    s = "";
    for slice in cleaned.codepoint_slices_reversed():
        s += String(slice);
    print(t"'{s}'"); # 'yduolC yltraP ,C5.02- :1 yaD'
```

`cleaned[byte=index]` is required. `cleaned[index]` does not compile. A byte index is O(1). A `codepoint=` index walks the text, and `grapheme=` does too. Grapheme indexing does not take a slice.

## 7. Slice a field

A slice `n:m` selects bytes `n` through `m - 1`. `n:` runs to the end. `:m` starts at the beginning. The result is a view, not a copy.

```mojo
    print(t"byte prefix:     '{cleaned[byte=:5]}'"); # 'Day 1'
    print(t"byte suffix:     '{cleaned[byte=cleaned.byte_length() - 5:]}'"); # 'loudy'
    print(t"byte substring:  '{cleaned[byte=2:10]}'"); # 'y 1: -20'
```

## 8. Search

`in` answers whether the text contains a substring. `find` returns the byte index of the first match, or `-1` when it is missing. `startswith` and `endswith` test the ends.

```mojo
    print(t"contains 'Day': {'Day' in cleaned}"); # True
    print(t"position of ':': {cleaned.find(':')}"); # 5
    print(t"position of '!': {cleaned.find('!')}"); # -1
    print(t"starts with 'Day': {cleaned.startswith('Day')}"); # True
    print(t"ends with 'Cloudy': {cleaned.endswith('Cloudy')}"); # True

    var pos = cleaned.find(":");
    if pos != -1:
        print(t"label: '{cleaned[byte=:pos]}'"); # 'Day 1'
    else:
        print("No label");

    if "!" in cleaned:
        print("Found");
    else:
        print("Not found");
```

Use `in` when you only need yes or no. Use `find` when you need the index.

## 9. Empty strings

An empty string is false. Any other string is true.

```mojo
    if "":
        print("Won't be printed");
    elif "🔥":
        print("This is printed");

    if cleaned:
        print(t"Processing: '{cleaned}'");
```

## 10. Replace

`replace` returns a new string and changes every match. The second call also removes the `C` in `Cloudy`.

```mojo
    var standardized = cleaned.replace("Partly Cloudy", "Cloudy");
    var no_units = standardized.replace("C", "");
    print(standardized); # Day 1: -20.5C, Cloudy
    print(no_units); # Day 1: -20.5, loudy
```

`removeprefix` and `removesuffix` remove a substring only at that end. `count` tells you how many non-overlapping copies are present.

## 11. Split and change case

`split(", ")` returns views into `cleaned`, so `cleaned` has to stay alive while you use `parts`.

```mojo
    var parts = cleaned.split(", ");
    print(t"split on ', ': {parts}"); # [Day 1: -20.5C, Partly Cloudy]
    print(t"count: {len(parts)}"); # 2

    print(parts[1].lower()); # partly cloudy
    print(parts[1].upper()); # PARTLY CLOUDY
    print(t"'cloudy' matches: {'cloudy' in cleaned.lower()}"); # True
```

`lower` and `upper` return new strings and affect ASCII letters. `split()` with no separator splits on whitespace.

## 12. Final code

Save this as `string_tour.mojo` in this folder.

```mojo
from std.pathlib import Path;

def main() raises:
    var log = Path("src/downloads/temp_log.txt");
    var string: String = String(log.read_text().splitlines()[0]);
    print(t"string: '{string}'");

    var length = string.byte_length();
    print(t"{string} length: {length}"); # 32

    print([String(t"'{slice}'") for slice in string.codepoint_slices()]);

    var joined = "".join([slice for slice in string.codepoint_slices()]);
    print(t"'{joined}'"); # '  Day 1: -20.5C, Partly Cloudy  '

    joined = "".join(
        [slice for slice in string.codepoint_slices() if " " not in slice]
    );
    print(t"'{joined}'"); # 'Day1:-20.5C,PartlyCloudy'

    var hello: String = "Hello";
    joined = ", ".join([slice for slice in hello.codepoint_slices()]);
    print(t"'{joined}'"); # 'H, e, l, l, o'

    var cleaned = string.strip();
    print(t"cleaned: '{cleaned}', bytes: {cleaned.byte_length()}");

    var flagged = "**" + cleaned + "**";
    print(t"'{flagged.strip("*")}'"); # 'Day 1: -20.5C, Partly Cloudy'

    var s: String = "";

    for index in reversed(range(cleaned.byte_length())):
        s = s + String(cleaned[byte=index]);
    print(t"Reversed bytes: '{s}'"); # 'yduolC yltraP ,C5.02- :1 yaD'

    s = "";
    for slice in cleaned.codepoint_slices_reversed():
        s += String(slice);
    print(t"'{s}'"); # 'yduolC yltraP ,C5.02- :1 yaD'

    print(t"byte prefix:     '{cleaned[byte=:5]}'"); # 'Day 1'
    print(t"byte suffix:     '{cleaned[byte=cleaned.byte_length() - 5:]}'"); # 'loudy'
    print(t"byte substring:  '{cleaned[byte=2:10]}'"); # 'y 1: -20'

    print(t"contains 'Day': {'Day' in cleaned}"); # True
    print(t"position of ':': {cleaned.find(':')}"); # 5
    print(t"position of '!': {cleaned.find('!')}"); # -1
    print(t"starts with 'Day': {cleaned.startswith('Day')}"); # True
    print(t"ends with 'Cloudy': {cleaned.endswith('Cloudy')}"); # True

    var pos = cleaned.find(":");
    if pos != -1:
        print(t"label: '{cleaned[byte=:pos]}'"); # 'Day 1'
    else:
        print("No label");

    if "!" in cleaned:
        print("Found");
    else:
        print("Not found");

    if "":
        print("Won't be printed");
    elif "🔥":
        print("This is printed");

    if cleaned:
        print(t"Processing: '{cleaned}'");

    var standardized = cleaned.replace("Partly Cloudy", "Cloudy");
    var no_units = standardized.replace("C", "");
    print(standardized); # Day 1: -20.5C, Cloudy
    print(no_units); # Day 1: -20.5, loudy

    var parts = cleaned.split(", ");
    print(t"split on ', ': {parts}"); # [Day 1: -20.5C, Partly Cloudy]
    print(t"count: {len(parts)}"); # 2

    print(parts[1].lower()); # partly cloudy
    print(parts[1].upper()); # PARTLY CLOUDY
    print(t"'cloudy' matches: {'cloudy' in cleaned.lower()}"); # True
```

## Topics covered

`byte_length`, `codepoint_slices`, `join`, `strip`, reversing, `byte=` slices, `find`, `in`, `replace`, `split`, `lower`, and `upper`.
