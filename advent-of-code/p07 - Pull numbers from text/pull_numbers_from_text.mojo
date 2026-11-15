from std.pathlib import Path;

# Scan for the numbers
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

def main() raises:
    var log = Path("src/downloads/temp_log.txt");
    var text = log.read_text();
    var lines = text.splitlines();
    print(text);

    var found = extract_ints(text);
    print(t"found: {found}");

    # Which numbers do you want?
    var day_numbers = List[Int]();
    for i in range(0, len(found), 3):
        day_numbers.append(found[i]);
    print(t"day numbers: {day_numbers}");

    # Check your work
    var expected = [i for i in range(1, len(lines) + 1)];
    print(t"complete, in order: {day_numbers == expected}");  # True