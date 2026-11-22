def calculate_average(temps: List[Float64]) raises -> Float64:
    if len(temps) == 0:
        raise Error(t"No temperature data");
    var total = 0.0;
    for temp in temps:
        total += temp;
    return total / Float64(len(temps));

def classify(avg: Float64) -> String:
    if avg > 25.0:
        return "Hot week";
    elif avg > 20.0:
        return "Comfortable week";
    else:
        return "Cool week";

def analyze(temps: List[Float64]):
    print("Recorded", len(temps), "temperatures");
    for index in range(len(temps)):
        print(t"Day {index + 1}: {temps[index]}");
    try:
        var avg = calculate_average(temps);
        print(t"Average temperature: {avg}");
        print("Status:", classify(avg));
    except e:
        print(t"Error: {e}");

def main():
    print("Temperature Analyzer");
    var week: List[Float64] = [20.5, 22.3, 19.8, 25.1];
    analyze(week);
    var nothing: List[Float64] = [];
    analyze(nothing);
