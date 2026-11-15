from std.pathlib import Path

def main() raises:
    # Build a string
    var log = Path("src/downloads/temp_log.txt");
    var string: String = String(log.read_text().splitlines()[0]);
    print(t"string: {string}");

    # Measure the input
    var length = string.byte_length();
    print(t"length: {length}");

    # String iteration
    print([String(t"'{slices}'") for slices in string.codepoint_slices()]);

    # Join slices from lists
    var joined = "".join([slice for slice in string.codepoint_slices()]);
    print(t"joined: {joined}");

    # Clean up the edges
    var cleaned = string.strip();
    print(t"cleaned: '{cleaned}', bytes: {cleaned.byte_length()}");

    var flagged = "**" + cleaned + "**";
    print(t"flagged: {flagged.strip("*")}");

    var s: String = "";
    # construct and reverse the non-inclusive range
    for index in reversed(range(0, cleaned.byte_length())):
        s += String(cleaned[byte=index]);
    print(t"Reversed bytes: '{s}'");

    # use the reversed iterator
    s = "";
    for slice in cleaned.codepoint_slices_reversed():
        s += String(slice);
    print(t"'{s}'");

    # O(1) byte slicing to a view
    print(t"byte prefix: '{cleaned[byte=:5]}'");
    print(t"byte suffix: '{cleaned[byte=cleaned.byte_length()-5:]}'");
    print(t"byte slice: '{cleaned[byte=2:10]}'");

    # Contains
    print(t"contains 'Day': {'Day' in cleaned}");

    # Position
    print(t"position of ':' : {cleaned.find(':')}");
    print(t"position of '!': {cleaned.find('!')}");  # None, not found

    # Start and end
    print(t"starts with 'Day': {cleaned.startswith('Day')}");
    print(t"ends with 'Cloudy': {cleaned.endswith('Cloudy')}");

    if "!" in cleaned:
        print(t"Found");
    else:
        print("Not found");

    # Test for empty strings
    if "":
        print("Won't be printed");
    elif "🔥":
        print("This is printed");
    
    # Use string truthiness when you care whether text is empty
    if cleaned:
        # There's something to process.
        print(t"Processing: '{cleaned}'");
    
    # Replace substrings
    var standardized = cleaned.replace("Partly Cloudy", "Cloudy");
    var no_units = standardized.replace("C", "");
    print(t"Standardized: '{standardized}', no units: '{no_units}'");

    # Split the line into fields
    var parts = cleaned.split(", ");
    print(t"split on ', ': {parts}");
    print(t"count: {len(parts)}");

    # Normalize for matching
    print(parts[1].lower());
    print(parts[1].upper());
    print(t"'cloudy' matches: {'cloudy' in parts[1].lower()}");