def main():
    # A bare bracket literal is a fixed-size array
    var fixed = [1, 2, 3];     # Array[Int, 3]
    fixed[0] = 10;             # elements can change
    print(len(fixed), fixed[0], fixed[2]);
    var zeros = Array[Float64, 4](fill=0.0);
    print(len(zeros), zeros[3]);

    # Annotate to get growable List.
    var nums: List[Int] = [2, 3, 5];
    nums.append(7);
    nums.append(11);
    print("popped:", nums.pop());
    print(nums);               # List prints its elements
    print("len:", len(nums), "last:", nums[len(nums) - 1]);

    var inferred: List = [1, 2, 3]; # element type inferred: List[Int]
    var names = List[String]();     # empty, typed via parameter
    names.append("Ada");
    names.append("Grace");
    print(inferred, names);

    # Slicing still works (no negative indexes)
    print(nums[1:3]);
    print(nums[len(nums) - 2:]);

    # Comprehensions
    var squares = [x * x for x in nums];
    var evens = [x for x in range(10) if x % 2 == 0];
    print(squares, evens);

    # Iterating: read-only by default, `ref` to modify in place
    for n in nums:
        print(n, end=" ");
    print();
    for ref n in nums:
        n *= 10;
    print(nums);

    # Membership, insert, extend
    print(30 in nums, 4 in nums);
    nums.insert(0, 1);
    nums.extend([99, 100]);
    print(nums);

    # Lists are not implicitly copied
    var backup = nums.copy();
    backup.append(-1);
    print(len(nums), len(backup));

    var digits: List[Int] = [x * x for x in range(10)];
    for ref num in digits:
        num *= 2;
    print(digits);




