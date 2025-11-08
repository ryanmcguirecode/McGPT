def test_hello_default():
    assert hello() == "Hello, world!"


def test_hello_custom():
    assert hello("Ryan") == "Hello, Ryan!"


def hello(name: str = "world") -> str:
    return f"Hello, {name}!"
