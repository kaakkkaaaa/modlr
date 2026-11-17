from std.collections import Set

def find_index(items: List[Int], target: Int) -> Optional[Int]:
    for i in range(len(items)):
        if items[i] == target:
            return i;
    return None;

def main() raises:
    # ----- Dict -----
    var capitals: Dict[String, String] = {
        "California": "Sacramento",
        "Hawaii": "Honolulu",
    };
    capitals["Oregon"] = "Salem"   # insert or update
    print(len(capitals), capitals["Hawaii"]);    # [] raises if key missing
    print("Texas" in capitals);
    print(capitals.get("Texas", "unknown"));     # no raise, returns default

    for state in capitals:        # keys, in insertion order
        print(state, end="; ");
    print();
    for item in capitals.items(): # key/value entries
        print(t"{item.value}, {item.key}");
    
    var removed = capitals.pop("Hawaii");
    print("removed", removed, "->", len(capitals), "left");

    var empty_scores: Dict[String, Int] = {};
    empty_scores["ada"] = 1;
    print(empty_scores);

    # ----- Set -----
    var i_like  = {"sushi", "ice cream", "tacos", "pho"};
    var you_like = {"burgers", "tacos", "salad", "ice cream"};
    print(i_like.intersection(you_like));
    var nums: Set[Int] = {1, 2, 3};
    nums.add(3);
    nums.add(4);
    print(len(nums), 2 in nums);
    var no_tags = Set[String]();
    print(len(no_tags));

    # ----- Optional -----
    var maybe: Optional[Int] = None;
    print(maybe.or_else(0));
    maybe = 42;
    if maybe:
        print("value:", maybe.value());
    if maybe is not None:
        print("not None");
    
    var greeting: Optional[String] = None;
    print(greeting.or_else("Hello"));

    var words : String = "the cat and the hat";
    var counts : Dict[String, Int] = {};
    for word in words.split():
        var key = String(word);
        counts[key] = counts.get(key, 0) + 1;
    print("the: ", counts.get("the", 0));
    print("cat: ", counts.get("cat", 0));
    print("and: ", counts.get("and", 0));
    print("hat: ", counts.get("hat", 0));

    var items: List[Int] = [10, 20, 30, 40];
    print(find_index(items, 30).or_else(-1));    # 2
    print(find_index(items, 99).or_else(-1));    # -1

