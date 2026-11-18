# p08 - Count and look up values

Step-by-step notes for [Count and look up values](https://advent.mojolang.org/count_and_look_up_values.html). The program in this folder is `count_and_look_up_values.mojo`.

It downloads the station reports, splits each line into a station id and a day, counts how often each station reported, finds the stations that stayed silent, and then looks for missing station-and-day pairs.

## Run

`src/downloads/station_reports.txt` is a path relative to this folder. The program downloads that file before it reads it.

```bash
cd "p08 - Count and look up values"
mojo count_and_look_up_values.mojo
```

Expected output:

```text
Downloaded file to  src/downloads/station_reports.txt
15 readings
station 1 has 5 readings
station 3 has 4 readings
station 5 has 6 readings
Station 1 reported? Yes
Station 2 reported? No
Station 3 reported? Yes
Station 4 reported? No
Station 5 reported? Yes
silent stations: {2, 4}
station 3 on day 0: 2
station 1 on day 0: 2
station 5 on day 0: 1
station 3 on day 1: 1
station 1 on day 2: 3
station 5 on day 1: 2
station 5 on day 2: 3
station 3 on day 2: 1
missing days: [Station 1 - Day 1]
```

A set has no order, so `silent stations` may print `{4, 2}` instead. The per-day lines follow the order each `(station, day)` pair first appears in the file.

## 1. Download and split the reports

`download` uses Python's `urllib.request` to save the lesson file. `read_text()` and `Int(...)` can fail, so `main()` is marked `raises`.

Each line looks like `Station 3, Day 0, 06:00, -22.3C, Clear`. Split on `", "` to get the fields, then split the station and day fields on a space and keep the number.

```mojo
from std.python import Python
from std.pathlib import Path
from std.collections import Set

def download(url: String, dest: String) raises:
    var request = Python.import_module("urllib.request");
    request.urlretrieve(url, dest);

def main() raises:
    var url = "https://advent.mojolang.org/downloads/station_reports.txt";
    var dest = "src/downloads/station_reports.txt";
    download(url, dest);
    print("Downloaded file to ", dest);

    var log = Path("src/downloads/station_reports.txt");
    var lines = log.read_text().splitlines();

    var reported_ids = List[Int]();
    var days = List[Int]();

    for line in lines:
        var fields = line.split(", ");
        reported_ids.append(Int(fields[0].split(" ")[1]));
        days.append(Int(fields[1].split(" ")[1]));
    print(t"{len(reported_ids)} readings");
```

The time, temperature, and conditions stay in `fields`. This page only needs the station id and the day. Fifteen lines produce fifteen readings.

## 2. Tally with a dictionary

`counts.get(id, 0)` returns the stored count, or `0` the first time a station appears. Adding `1` is the tally.

`items()` walks the pairs, but a dictionary does not keep keys in numeric order. Copy them into a list and sort that list by station id before printing.

```mojo
    var counts = Dict[Int, Int]();
    for id in reported_ids:
        counts[id] = counts.get(id, 0) + 1;

    var pairs = List[Tuple[Int, Int]]();
    for entry in counts.items():
        pairs.append((entry.key, entry.value));

    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if pairs[j][0] < pairs[i][0]:
                var temp = pairs[i];
                pairs[i] = pairs[j];
                pairs[j] = temp;

    for pair in pairs:
        print(t"station {pair[0]} has {pair[1]} readings");
```

Station 1 reported five times, station 3 four times, and station 5 six times. Stations 2 and 4 never appear.

## 3. Look up with a set

`Set` is not in the prelude, so it is imported from `std.collections`. `add` keeps each station id once. `station in reported` answers membership without scanning the original list.

```mojo
    var reported: Set[Int] = {};
    for id in reported_ids:
        reported.add(id);

    for station in range(1, 6):
        print(
            t"Station {station} reported?"
            t"{' Yes' if station in reported else ' No'}"
        );
```

The second t-string supplies the space before `Yes` or `No`. Stations 1, 3, and 5 reported. Stations 2 and 4 did not.

## 4. Who went silent

Subtract the stations that reported from the roster you expected. Set difference keeps everything in `expected` that is not in `reported`.

```mojo
    var expected = Set[Int](1, 2, 3, 4, 5);
    var missing = expected - reported;
    print(t"silent stations: {missing}");
```

`{1, 2, 3}` is a set. `{1: 2, 3: 4}` is a dictionary. The colon is what tells them apart.

## 5. Count each station and day

One id is not enough to count readings per station per day. A tuple of two `Int`s is hashable, so it can be a dictionary key. Unpacking `entry.key` names the two halves.

```mojo
    var per_day = Dict[Tuple[Int, Int], Int]();
    for i in range(len(reported_ids)):
        var key = (reported_ids[i], days[i]);
        per_day[key] = per_day.get(key, 0) + 1;

    for entry in per_day.items():
        var station, day = entry.key;
        print(t"station {station} on day {day}: {entry.value}");
```

## 6. Find the missing days

Stations 2 and 4 never reported. Among the stations that did, check days `0`, `1`, and `2`. Build a set of the pairs that showed up, then a comprehension keeps the combinations that are absent.

```mojo
    var per_day_set = Set[Tuple[Int, Int]]();
    for i in range(len(reported_ids)):
        per_day_set.add((reported_ids[i], days[i]));

    var missing_days = [
        String(t"Station {station} - Day {day}")
        for station in [1, 3, 5]
        for day in range(3)
        if (station, day) not in per_day_set
    ];

    print(t"missing days: {missing_days}");
```

Station 1 never reported on day 1. That is the only missing pair among stations 1, 3, and 5.

## Topics covered

Downloading with `urllib.request`, splitting report fields, dictionaries as counters, `get()` with a default, sorting `items()` by key, sets for uniqueness and membership, set difference, tuples as compound keys, tuple unpacking, and a comprehension that filters missing combinations.
