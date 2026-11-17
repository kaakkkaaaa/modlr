def show(label: StaticString, s: StringSlice):
    print(
        label,
        "bytes=", s.byte_length(),
        "codepoints=", s.count_codepoints(),
        "graphemes=", s.count_graphemes(),
    );

def main():
    var s: String = "Testing";
    s += " Mojo strings";
    print(s);

    # Building string from other values
    print("Items: " + String(5));
    print(String("Items: ", 5, ", ratio: ", 0.5));

    # Three ways to measure text
    show("ascii  ", "hello");
    show("cafe   ", "café");
    show("wave   ", "👋🏽");

    # Indexing is explicit about units
    var word = "café";
    print(word[byte=0]);            # first byte, as a StringSlice
    print(word[codepoint=3]);       # fourth codepoint: café -> é
    print(word[codepoint=0:3]);     # first 3 codepoints: café -> caf

    # Iterate codepoints, not bytes
    for ch in word.codepoint_slices():
        print(ch, end="|");
    print();

    # Template strings (t-string)
    var count = 3;
    var items = "apples";
    print(t"Give me {count} {items}");
    print(t"The answer is {count * 14}");
    var built = String(t"{count}-{items}");
    print(built);

    # format()
    print("{0} {1} {0}".format("Mojo", 1.125));

    # Raw strings and escapes
    print(r"C:\new\table");
    print("tab:\tend", "\u20AC");

    # Multi-line literal
    var poem = """Roses are red,
Mojo is fast.""";
    print(poem);

    # Common methods
    var csv = "a,b,c";
    var parts = csv.split(",");
    print(len(parts), parts[1]);
    print(csv.upper(), csv.find("b"), "b" in csv, csv.startswith("a"));
    print("ha" * 3, "Zebra" < "ant");






