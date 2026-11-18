def describe_status(status_code: Int) -> String:
    match status_code:
        case 200:
            return "OK";
        case 301 | 302:
            return "Redirect";
        case 404:
            return "Not found";
        case code if code >= 500:
            return String(t"Server error {code}");
        case _:
            return "Unexpected";

def forecast(temp_celcius: Optional[Int]) -> String:
    match temp_celcius:
        case .Some(temp) if temp > 20:
            return "warm";
        case .Some(_):
            return "cool";
        case .None:
            return "no reading";

def next_reading(mut readings: List[Int]) -> Optional[Int]:
    if readings:
        return readings.pop();
    return None;

def describe_temp(temp_celcius: Int) -> String:
    match temp_celcius:
        case t if t <= 0:
            return "It's freezing!";
        case t if t < 20:
            return "It's cool.";
        case t if t < 30:
            return "It's warm.";
        case _:
            return "It's hot!";

def quadrant(p: Tuple[Int, Int]) -> String:
    match p:
        case (0, 0):
            return "origin";
        case (0, _):
            return "y axis";
        case (_, 0):
            return "x axis";
        case (x, y) if x > 0 and y > 0:
            return "Q1";
        case (x, y) if x < 0 and y > 0:
            return "Q2";
        case (x, y) if x < 0 and y < 0:
            return "Q3";
        case _:
            return "Q4";

def main():
    for code in [200, 302, 404, 503, 418]:
        print(code, describe_status(code));
    
    print(forecast(25), forecast(12), forecast(None));

    # if case: test one pattern and bind its contents
    var reading: Optional[Int] = 25;
    if case .Some(temp) = reading:
        print("The temperature is", temp);
    else:
        print("No reading");
    
    # while case: loop while the pattern keeps matching
    var readings: List[Int] = [18, 22, 25];
    while case .Some(temp) = next_reading(readings):
        print(temp, end=", ");
    else:
        print("done");
    
    # match on a tuple
    var point = (0, 5);
    match point:
        case (0, 0):
            print("origin");
        case (0, y):
            print("on the y axis at", y);
        case (x, y):
            print("at", x, y);

    # match on a string
    var command = String("stop");
    match command:
        case "go":
            print("going");
        case "stop" | "halt":
            print("stopping");
        case _:
            print("unknown command");

    # Exercise: Temperature chain as a match
    print(describe_temp(-5), describe_temp(0), describe_temp(12), describe_temp(25), describe_temp(35));

    # Exercise: Quadrant with tuple patterns and guards
    print(quadrant((0, 0)), quadrant((0, 5)), quadrant((5, 0)), quadrant((5, 5)), quadrant((-5, 5)), quadrant((-5, -5)));