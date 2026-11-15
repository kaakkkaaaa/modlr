from std.pathlib import Path
from std.os import remove

def main() raises:
    # Read the sensor log
    var log = Path("src/downloads/temp_log.txt");
    print(log.read_text());

    try:
        var log = Path("src/downloads/temp.txt").read_text();
    except e:
        print(t"Error: {e}");

    print(t"\nExists: {log.exists()}");
    print(t"File: {log.is_file()}");
    print(t"Dir: {log.is_dir()}");

    # Split into lines
    var text = log.read_text();
    var lines = text.splitlines();
    print(t"\nGot {len(lines)} lines");   # Got 10 lines

    # Create a report
    var report = Path("report.txt");

    report.write_text(
        String(
            "North Pole Temperature Report",
            "==============================",
            "",
            "Input: temp_log.txt",
            "Status: Received",
            "",
            "Waiting for analysis...",
            "",
            sep="\n",
        )
    );

    # Append content
    with open(report, "a") as f:
        var value = -52.3;
        f.write(t"\nLate reading: {Float64(value)} °C");
    print(report.read_text());

    # Clean up
    try:
        if report.exists():
            remove(report);
    except e:
        print(t"File removal failed: {e}");
    
    print(t"After cleanup: {report.exists()}");
