from std.pathlib import Path

def main() raises:
    # Splitting lines
    var log = Path("src/downloads/temp_log.txt");
    var text = log.read_text();
    var lines = text.splitlines();
    print(t"Got {len(lines)} lines");   # Got 10 lines

    var split_on_newline = text.split("\n");
    print(t"split got {len(split_on_newline)} lines");

    # Tidy your input
    var cleaned_lines: List[String] = [];
    for line in lines:
        var cleaned = line.strip();
        if not cleaned:
            continue;
        cleaned_lines.append(String(cleaned));
        print(t"cleaned: {cleaned}");
    
    # Extract the fields
    var first = cleaned_lines[0];
    var reading = first.split(": ")[1].split(", ");
    var temp_field = reading[0];
    var conditions = reading[1];
    print(t"temp field: '{temp_field}', conditions: '{conditions}'");

    # Type conversions
    try:
        _ = Float64(temp_field);
    except e:
        print(t"Float64('{temp_field}') raised: {e}");
    
    var temp = Float64(temp_field.removesuffix("C"));
    print(t"temp: {temp}");

    # Collect all readings
    var temps: List[Float64] = [];

    for line in cleaned_lines:
        var fields = line.split(": ")[1].split(", ");
        temps.append(Float64(fields[0].removesuffix("C")));
    
    print(t"Parsed {len(temps)} temperatures: {temps}");

    # Handle invalid lines
    # Simulate one more reading coming in from a flaky sensor
    var extended_lines = cleaned_lines.copy();
    extended_lines.append("Day 11: ERROR, Sensor Fault");

    temps = [];
    var rejected: List[String] = [];

    for line in extended_lines:
        var fields = line.split(": ")[1].split(", ");

        try:
            temps.append(Float64(fields[0].removesuffix("C")));
        except e:
            rejected.append(line);
    
    print(t"Parsed {len(temps)} temperatures: {temps}");
    print(t"Rejected {len(rejected)} : {rejected}");

    for line in extended_lines:
        var fields = line.split(": ")[1].split(", ");

        try:
            _ = Float64(fields[0].removesuffix("C"));
        except e:
            print(t"Rejected: '{line}' raised: {e}");