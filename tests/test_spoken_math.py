import pytest

from arietta_voice.config import default_config
from arietta_voice.spoken_math import (
    MathError,
    evaluate,
    parse_conversion,
    parse_expression,
    say_number,
)
from arietta_voice.tools import ToolRegistry
from arietta_voice.tools.builtin.calculate import CalculateTool
from arietta_voice.tools.registry import run_tool


@pytest.mark.parametrize(
    "text,spoken,value",
    [
        ("What's seventeen times twenty three?", "17 times 23", 391),
        ("what is 17 x 23", "17 times 23", 391),
        ("17×23", "17 times 23", 391),
        ("15% of 80", "15 percent of 80", 12),
        ("what is fifteen percent of eighty", "15 percent of 80", 12),
        ("square root of 144", "the square root of 144", 12),
        ("the cube root of 27", "the cube root of 27", 3),
        ("two to the power of ten", "2 to the power of 10", 1024),
        ("two to the tenth", "2 to the power of 10", 1024),
        ("2 to the 10th", "2 to the power of 10", 1024),
        ("5 squared", "5 squared", 25),
        ("three cubed", "3 cubed", 27),
        ("100 divided by 8", "100 divided by 8", 12.5),
        ("seven over two", "7 divided by 2", 3.5),
        ("one thousand two hundred and five plus 6", "1,205 plus 6", 1211),
        ("1.5 million times 2", "1,500,000 times 2", 3_000_000),
        ("three point one four times two", "3.14 times 2", 6.28),
        ("minus five plus three", "minus 5 plus 3", -2),
        ("what's 1,234 + 5,678", "1,234 plus 5,678", 6912),
        ("what does 12 times 12 equal", "12 times 12", 144),
        ("2 plus 3 times 4", "2 plus 3 times 4", 14),
        ("ten mod three", "10 modulo 3", 1),
    ],
)
def test_spoken_arithmetic(text, spoken, value):
    parsed = parse_expression(text)
    assert parsed is not None, text
    assert parsed.spoken == spoken
    assert evaluate(parsed.python) == pytest.approx(value)


@pytest.mark.parametrize(
    "text",
    [
        "what is the capital of France",
        "what time is it",
        "two plus",
        "seventeen",
        "set a timer for ten minutes",
        "how many minutes are left",
        "tell me a joke",
        "",
    ],
)
def test_other_requests_are_not_arithmetic(text):
    assert parse_expression(text) is None


@pytest.mark.parametrize(
    "expression,message",
    [
        ("10 / 0", "divide by zero"),
        ("sqrt(-1)", "real square root"),
        ("9 ** 9 ** 9", "too big"),
        ("10 ** 101", "too big"),
        ("__import__('os').system('true')", "can work out"),
        ("open('x')", "can work out"),
        ("(1).__class__", "can work out"),
        ("[1, 2]", "can work out"),
        ("'a' * 3", "can work out"),
        ("x + 1", "can work out"),
        ("1 +", "can work out"),
        ("1" * 201, "too long"),
    ],
)
def test_only_arithmetic_is_ever_evaluated(expression, message):
    with pytest.raises(MathError, match=message):
        evaluate(expression)


def test_model_style_expressions():
    assert evaluate("17*23") == 391
    assert evaluate("15% * 80") == 12
    assert evaluate("2^10") == 1024
    assert evaluate("sqrt(144) + abs(-1)") == 13
    assert evaluate("7 // 2") == 3


@pytest.mark.parametrize(
    "value,spoken",
    [
        (391, "391"),
        (1024.0, "1,024"),
        (12.5, "12.5"),
        (100 / 7, "14.29"),
        (7 / 9, "0.778"),
        (0.000123, "0.000123"),
        (-2, "-2"),
        (1234567.891, "1,234,567.9"),
    ],
)
def test_numbers_are_said_simply(value, spoken):
    assert say_number(value) == spoken


@pytest.mark.parametrize(
    "text,spoken",
    [
        ("convert 5 miles to kilometers", "5 miles is about 8.05 kilometers."),
        ("five miles to kilometers", "5 miles is about 8.05 kilometers."),
        (
            "what's 70 fahrenheit in celsius",
            "70 degrees Fahrenheit is about 21.11 degrees Celsius.",
        ),
        (
            "convert 100 degrees celsius to fahrenheit",
            "100 degrees Celsius is 212 degrees Fahrenheit.",
        ),
        ("how many cups are in a liter", "1 liter is about 4.23 cups."),
        ("how many feet in a mile", "1 mile is 5,280 feet."),
        ("how many teaspoons in a tablespoon", "1 tablespoon is 3 teaspoons."),
        ("how many inches are in 3 feet", "3 feet is 36 inches."),
        ("how many kilometers is 10 miles", "10 miles is about 16.09 kilometers."),
        ("what is 1 inch in centimeters", "1 inch is 2.54 centimeters."),
        ("100 km/h in mph", "100 kilometers per hour is about 62.14 miles per hour."),
        ("how many seconds in a day", "1 day is 86,400 seconds."),
    ],
)
def test_unit_conversions(text, spoken):
    conversion = parse_conversion(text)
    assert conversion is not None, text
    assert conversion.spoken() == spoken


@pytest.mark.parametrize(
    "text",
    [
        "convert 5 miles to pounds",
        "how many timers do I have",
        "5 miles to miles",
        "what's 5 bananas",
    ],
)
def test_conversions_need_two_matching_units(text):
    assert parse_conversion(text) is None


@pytest.fixture
def registry(tmp_path):
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    return ToolRegistry(config)


@pytest.mark.parametrize(
    "text,speak",
    [
        ("What's seventeen times twenty three?", "17 times 23 is 391."),
        ("what is 15% of 80", "15 percent of 80 is 12."),
        ("square root of 2", "The square root of 2 is 1.41."),
        ("what's 10 divided by 0", "You can't divide by zero."),
        ("how many cups are in a liter", "1 liter is about 4.23 cups."),
    ],
)
def test_math_is_answered_instantly(registry, text, speak):
    hit = registry.fast_path(text)
    assert (hit.name, hit.result.speak) == ("calculate", speak)


@pytest.mark.parametrize("text", ["What time is it?", "set a 5 minute timer", "who are you"])
def test_other_instant_answers_still_win(registry, text):
    assert registry.fast_path(text).name != "calculate"


def test_the_model_can_call_it_with_an_expression():
    tool = CalculateTool()
    assert run_tool(tool, {"expression": "17 * 23"}).speak == "17 times 23 is 391."
    assert run_tool(tool, {"expression": "sqrt(144) + 1"}).speak == "That's 13."
    assert run_tool(tool, {"query": "2**10"}).speak == "That's 1,024."
    assert run_tool(tool, {"expression": "5 miles to km"}).speak == (
        "5 miles is about 8.05 kilometers."
    )
    refused = run_tool(tool, {"expression": "__import__('os')"})
    assert refused.speak == "That isn't an expression I can work out."
    assert tool.speaks_for_itself
