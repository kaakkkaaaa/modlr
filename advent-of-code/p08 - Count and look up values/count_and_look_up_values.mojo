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

    # The situation report
    var log = Path("src/downloads/station_reports.txt");
    var lines = log.read_text().splitlines();

    var reported_ids = List[Int]();
    var days = List[Int]();

    for line in lines:
        # ['Station 3', 'Day 0', '06:00', '-18.2C', 'Clear']
        var fields = line.split(", ");
        reported_ids.append(Int(fields[0].split(" ")[1]));
        days.append(Int(fields[1].split(" ")[1]));
        # fields[2] is the time, fields[3] the temperature, fields[4] the
        # conditions. This page only needs the station id and the day.
    print(t"{len(reported_ids)} readings"); # 15 readings

    # Tally with a dictionary
    var counts = Dict[Int, Int]();
    for id in reported_ids:
        counts[id] = counts.get(id, 0) + 1;
        # Key not found? Count defaults to 0.
    
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

    # Look up fast with a set
    var reported: Set[Int] = {};
    for id in reported_ids:
        reported.add(id);
    
    for station in range(1, 6):
        print(
            t"Station {station} reported?"
            t"{' Yes' if station in reported else ' No'}"
        );
    
    # Who went silent?
    var expected = Set[Int](1, 2, 3, 4, 5);
    var missing = expected - reported;
    print(t"silent stations: {missing}");

    # Tuples: Mojo's anonymous types
    var per_day = Dict[Tuple[Int, Int], Int]();
    for i in range(len(reported_ids)):
        var key = (reported_ids[i], days[i]);
        per_day[key] = per_day.get(key, 0) + 1;
    
    for entry in per_day.items():
        var station, day = entry.key;
        print(t"station {station} on day {day}: {entry.value}");
    
    # Comprehensions: smart data retrieval
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