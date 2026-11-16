def greet(name: String) -> String:
    """
    Returns a greeting for `name`.

    Args:
        name: Who to greet.
    Returns:
        The greeting text.
    """
    return "Hello, " + name + "!";

def farewell(name: String) -> String:
    """
    Return a farewell for `name`.

    Args:
        name: Who to farewell.
    Returns:
        The farewell text.
    """
    return "Goodbye, " + name + "!";

def main():
    print("Hello, World!");
    print(greet("Kaká")); # a trailing comment
    print(farewell(""))

    # Break long expressions inside (), [], or {}.
    var long_text = (
        "Adjacent string literals"
        " are joined into one string."
    );
    print(long_text);
    print("a", "b", 3, sep="-", end="!\n");
    print("a", "b", 3, sep=",", end="!\n");