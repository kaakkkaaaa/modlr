# p04 - Work with files

Step-by-step notes for [Work with files](https://advent.mojolang.org/work_with_files.html). You write `temp_log.mojo` yourself as you go. This folder does not include that program.

The program you end up with reads `temp_log.txt`, checks that the path is a file, splits it into lines, writes `report.txt`, appends one late reading, prints the report, and deletes it.

This chapter lives in the `advent-of-code` Pixi project. The environment files are in the parent folder, not in this one:

- `pixi.toml` installs Mojo
- `pixi.lock` pins that environment
- `.gitignore` keeps `.pixi/` out of git
- `.gitattributes` marks the lockfile as generated

Install the environment once from `advent-of-code`:

```bash
pixi install
```

## Run

After you have written `temp_log.mojo` in this folder, start a Pixi shell from `advent-of-code` and run it here. `temp_log.txt` and `report.txt` are paths relative to the current directory.

```bash
pixi shell
cd "p04 - Work with files"
mojo temp_log.mojo
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

Exists: True
File:   True
Dir:    False
Got 10 lines
North Pole Temperature Report
=============================

Input: temp_log.txt
Status: Received

Waiting for analysis...

Late reading: -52.3 °C
After cleanup, exists: False
```

`report.txt` is gone when the program finishes. The sensor log stays.

## 1. Grab the sensor log

`temp_log.txt` in this folder is the sample from the chapter. Leave its whitespace alone. Line 1 has two spaces at each end, line 3 and line 9 have leading spaces, and line 6 has extra spaces before `Foggy`. Later chapters measure those bytes.

The book loads `src/downloads/temp_log.txt`. Here the log sits next to the program, so the path is `temp_log.txt`.

## 2. Read the log

Create `temp_log.mojo`. `Path` is a filesystem path. `read_text()` opens the file, reads it as one `String`, and closes it. `main()` is marked `raises` because opening a missing file is an error.

```mojo
from std.pathlib import Path;

def main() raises:
    var log = Path("temp_log.txt");
    print(log.read_text()); # Day 1: -20.5C, Partly Cloudy, etc
```

Run it from this folder. The ten lines should match the file, including the blank line after the last reading. `print` adds that extra newline.

## 3. Check the path

Ask whether the path exists, and whether it is a file or a directory. All three return `Bool`.

```mojo
    print(t"Exists: {log.exists()}");
    print(t"File:   {log.is_file()}");
    print(t"Dir:    {log.is_dir()}");
```

With this log the answers are `True`, `True`, and `False`.

## 4. A missing file

Point `log` at a path that is not on disk:

```mojo
    var log = Path("path/to/nowhere");
```

Take `raises` off `main()`. Mojo refuses to compile, because `read_text()` can fail and nothing is allowed to raise.

Put `raises` back, then catch the failure:

```mojo
def main() raises:
    var log = Path("path/to/nowhere");

    try:
        print(log.read_text());
    except e:
        print(t"Error: {e}");
```

The program prints the error instead of stopping. Put `temp_log.txt` back when you are done.

## 5. Split into lines

Advent input is usually one record per line. `splitlines()` returns a `List` of views into `text`, so `text` has to stay alive while you use `lines`. It accepts `\n`, `\r\n`, and `\r`, and it does not add an empty entry for a final newline.

```mojo
    var text = log.read_text();
    var lines = text.splitlines();
    print(t"Got {len(lines)} lines"); # Got 10 lines
```

## 6. Write a report

`write_text()` creates the file, or replaces it if it already exists. It also opens and closes the file for you. `String(..., sep="\n")` joins those pieces with newlines. The last empty string is what leaves a trailing newline.

```mojo
    var report = Path("report.txt");

    report.write_text(
        String(
            "North Pole Temperature Report",
            "=============================",
            "",
            "Input: temp_log.txt",
            "Status: Received",
            "",
            "Waiting for analysis...",
            "",
            sep="\n",
        )
    );
```

Open `report.txt` before the next step if you want to see it without the late reading. The finished program deletes it.

## 7. Append a late reading

`open(path, mode)` returns a `FileHandle`. `"a"` appends. `"r"` reads, `"w"` truncates, and `"rw"` does both. `write()` accepts any `Writable` value, including a template string.

```mojo
    with open(report, "a") as f:
        var value = -52.3;
        f.write(t"\nLate reading: {Float64(value)} °C");
    print(report.read_text());
```

The new line is `-52.3 °C`. The `with` block closes the handle when it ends, including when the block raises.

A context manager is any type with `__enter__()` and `__exit__()`. There is no context-manager trait. Mojo calls those two methods around the block, and the name after `as` is the value `__enter__()` returns.

## 8. Delete the report

The sensor log is input, so it stays. The report is generated, so the program removes it. `remove()` raises when the delete fails. The `exists()` check avoids that for a missing file, and `try` / `except` covers a delete that fails for another reason.

```mojo
from std.os import remove;

try:
    if report.exists():
        remove(report);
except e:
    print(t"File removal failed: {e}");

print(t"After cleanup, exists: {report.exists()}");
```

The last line prints `After cleanup, exists: False`.

## 9. Final code

Save this as `temp_log.mojo` in this folder. The `remove` import sits at the top with `Path`.

```mojo
from std.pathlib import Path;
from std.os import remove;

def main() raises:
    var log = Path("temp_log.txt");
    print(log.read_text()); # Day 1: -20.5C, Partly Cloudy, etc

    print(t"Exists: {log.exists()}");
    print(t"File:   {log.is_file()}");
    print(t"Dir:    {log.is_dir()}");

    var text = log.read_text();
    var lines = text.splitlines();
    print(t"Got {len(lines)} lines"); # Got 10 lines

    var report = Path("report.txt");

    report.write_text(
        String(
            "North Pole Temperature Report",
            "=============================",
            "",
            "Input: temp_log.txt",
            "Status: Received",
            "",
            "Waiting for analysis...",
            "",
            sep="\n",
        )
    );

    with open(report, "a") as f:
        var value = -52.3;
        f.write(t"\nLate reading: {Float64(value)} °C");
    print(report.read_text());

    try:
        if report.exists():
            remove(report);
    except e:
        print(t"File removal failed: {e}");

    print(t"After cleanup, exists: {report.exists()}");
```

## Topics covered

`Path`, `read_text()`, `write_text()`, `splitlines()`, `open()` and file modes, `with`, and `remove()`.
