# p06 - Parse text into typed data

Step-by-step notes for [Parse text into typed data](https://advent.mojolang.org/parse_text.html). You write `parse_text.mojo` yourself as you go. This folder does not include that program.

The program you end up with reads the sensor log, splits it into lines, trims each line, pulls out the temperature and the weather, converts the temperatures to `Float64`, and keeps going when one line is not a number.

## Run

After you have written `parse_text.mojo` here, run it from this folder. `src/downloads/temp_log.txt` is a path relative to the current directory.

```bash
cd "p06 - Parse text into typed data"
mojo parse_text.mojo
```

`src/downloads/temp_log.txt` is the same sensor log as the previous chapters. Line 6 keeps the extra spaces before `Foggy`. Those spaces are inside the line, so `strip()` does not remove them.

Expected output:

```text
Got 10 lines
split got 11 lines
line: 'Day 1: -20.5C, Partly Cloudy'
line: 'Day 2: -22.3C, Snow'
line: 'Day 3: -19.8C, Clear'
line: 'Day 4: -25.1C, Partly Cloudy'
line: 'Day 5: -44.5C, BLIZZARD'
line: 'Day 6: -27.7C,   Foggy'
line: 'Day 7: -42.3C, partly cloudy'
line: 'Day 8: -28.2C, Clear'
line: 'Day 9: -23.0C, Snow'
line: 'Day 10: -36.5C, Partly Cloudy'
temp field: '-20.5C', conditions: 'Partly Cloudy'
Float64('-20.5C') raised: String is not convertible to float: '-20.5C'. The last character of '20.5C' should be a digit or dot to convert it to a float.
temp: -20.5
Parsed 10 temperatures: [-20.5, -22.3, -19.8, -25.1, -44.5, -27.7, -42.3, -28.2, -23.0, -36.5]
Parsed 10 temperatures: [-20.5, -22.3, -19.8, -25.1, -44.5, -27.7, -42.3, -28.2, -23.0, -36.5]
Rejected 1: [Day 11: ERROR, Sensor Fault]
Rejected 'Day 11: ERROR, Sensor Fault': String is not convertible to float: 'ERROR'. The first character of 'ERROR' should be a digit or dot to convert it to a float.
[Day 6: -27.7C,    Foggy]
```

## 1. Split the log

Create `parse_text.mojo`. `splitlines()` returns one entry per line and does not add an empty entry for the final newline. `read_text()` can fail, so `main()` is marked `raises`.

```mojo
from std.pathlib import Path;

def main() raises:
    var log = Path("src/downloads/temp_log.txt");
    var text = log.read_text();
    var lines = text.splitlines();
    print(t"Got {len(lines)} lines"); # 10
```

`split("\n")` keeps the empty piece created by that final newline, so the count is 11. Use `splitlines()` for line-oriented input.

```mojo
    var split_on_newline = text.split("\n");
    print(t"split got {len(split_on_newline)} lines"); # 11
```

## 2. Clean each line

`strip()` returns a view with whitespace removed from both ends. An empty view is false, so `continue` skips a blank line. `String(cleaned)` copies the text out of `text`. After that, `cleaned_lines` does not depend on `text` staying alive.

```mojo
    var cleaned_lines: List[String] = [];
    for line in lines:
        var cleaned = line.strip();
        if not cleaned:
            continue;
        cleaned_lines.append(String(cleaned));
        print(t"line: '{cleaned}'");
```

## 3. Pull the fields apart

Each cleaned line has a day, a temperature with a `C`, and the weather. Split on `": "` first, then split the remainder on `", "`. Each `split` returns views into the string you split.

```mojo
    var first = cleaned_lines[0];
    var reading = first.split(": ")[1].split(", ");
    var temp_field = reading[0]; # '-20.5C'
    var conditions = reading[1]; # 'Partly Cloudy'
    print(t"temp field: '{temp_field}', conditions: '{conditions}'");
```

## 4. Convert the temperature

`Float64(text)` raises when the text is not a number. The trailing `C` is not part of the number. `removesuffix("C")` removes that unit only when it is at the end.

```mojo
    try:
        _ = Float64(temp_field);
    except e:
        print(t"Float64('{temp_field}') raised: {e}");

    var temp = Float64(temp_field.removesuffix("C"));
    print(t"temp: {temp}"); # -20.5
```

## 5. Collect every reading

A `List[Float64]` accepts only `Float64` values. Repeat the same split and conversion for every cleaned line.

```mojo
    var temps: List[Float64] = [];

    for line in cleaned_lines:
        var fields = line.split(": ")[1].split(", ");
        temps.append(Float64(fields[0].removesuffix("C")));

    print(t"Parsed {len(temps)} temperatures: {temps}");
```

That prints 10 temperatures.

## 6. Keep a bad line from stopping the parse

`copy()` duplicates the list. Append one corrupt reading, then catch a failed conversion and save that line instead of stopping.

`except:` catches the error and discards it. `except e:` binds the error so you can print it. The rejected list stores `String` values, not views into the original line.

```mojo
    var extended_lines = cleaned_lines.copy();
    extended_lines.append("Day 11: ERROR, Sensor Fault");

    temps = [];
    var rejected: List[String] = [];

    for line in extended_lines:
        var fields = line.split(": ")[1].split(", ");

        try:
            temps.append(Float64(fields[0].removesuffix("C")));
        except:
            rejected.append(line);

    print(t"Parsed {len(temps)} temperatures: {temps}");
    print(t"Rejected {len(rejected)}: {rejected}");

    for line in extended_lines:
        var fields = line.split(": ")[1].split(", ");

        try:
            _ = Float64(fields[0].removesuffix("C"));
        except e:
            print(t"Rejected '{line}': {e}");
```

Ten temperatures parse. `Day 11: ERROR, Sensor Fault` is the rejected line.

## 7. A separator that is only a comma

Splitting line 6 on `","` leaves the spaces that sit after the comma. That is why the weather field still starts with spaces.

```mojo
    var row = cleaned_lines[5]; # 'Day 6: -27.7C,   Foggy'
    var columns = row.split(",");
    print(columns); # [Day 6: -27.7C,    Foggy]
```

## 8. Final code

Save this as `parse_text.mojo` in this folder.

```mojo
from std.pathlib import Path;

def main() raises:
    var log = Path("src/downloads/temp_log.txt");
    var text = log.read_text();
    var lines = text.splitlines();
    print(t"Got {len(lines)} lines"); # 10

    var split_on_newline = text.split("\n");
    print(t"split got {len(split_on_newline)} lines"); # 11

    var cleaned_lines: List[String] = [];
    for line in lines:
        var cleaned = line.strip();
        if not cleaned:
            continue;
        cleaned_lines.append(String(cleaned));
        print(t"line: '{cleaned}'");

    var first = cleaned_lines[0];
    var reading = first.split(": ")[1].split(", ");
    var temp_field = reading[0]; # '-20.5C'
    var conditions = reading[1]; # 'Partly Cloudy'
    print(t"temp field: '{temp_field}', conditions: '{conditions}'");

    try:
        _ = Float64(temp_field);
    except e:
        print(t"Float64('{temp_field}') raised: {e}");

    var temp = Float64(temp_field.removesuffix("C"));
    print(t"temp: {temp}"); # -20.5

    var temps: List[Float64] = [];

    for line in cleaned_lines:
        var fields = line.split(": ")[1].split(", ");
        temps.append(Float64(fields[0].removesuffix("C")));

    print(t"Parsed {len(temps)} temperatures: {temps}");

    var extended_lines = cleaned_lines.copy();
    extended_lines.append("Day 11: ERROR, Sensor Fault");

    temps = [];
    var rejected: List[String] = [];

    for line in extended_lines:
        var fields = line.split(": ")[1].split(", ");

        try:
            temps.append(Float64(fields[0].removesuffix("C")));
        except:
            rejected.append(line);

    print(t"Parsed {len(temps)} temperatures: {temps}");
    print(t"Rejected {len(rejected)}: {rejected}");

    for line in extended_lines:
        var fields = line.split(": ")[1].split(", ");

        try:
            _ = Float64(fields[0].removesuffix("C"));
        except e:
            print(t"Rejected '{line}': {e}");

    var row = cleaned_lines[5]; # 'Day 6: -27.7C,   Foggy'
    var columns = row.split(",");
    print(columns); # [Day 6: -27.7C,    Foggy]
```

## Topics covered

`splitlines`, `strip`, `continue`, copying a view into a `String`, chained `split`, `Float64` conversion, `removesuffix`, `List[Float64]`, and `try` / `except`.
