"""Arithmetic and unit conversion, read the way people say them.

"What's seventeen times twenty three?", "15% of 80", "square root of 144",
"convert 5 miles to kilometers", "how many cups are in a liter". Expressions are
evaluated by walking a parsed syntax tree that allows only numbers, the four
operations, powers, remainders and a few functions, never ``eval``.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass

from arietta_voice.timephrases import ONES, TENS

SCALES = {"thousand": 10**3, "million": 10**6, "billion": 10**9, "trillion": 10**12}
ORDINAL_POWERS = {"second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "tenth": 10}
LEAD_IN = re.compile(
    r"^(?:(?:hey|ok(?:ay)?|so|please|quick question|can you|could you|would you)\s+)*"
    r"(?:what(?:'s| is| are| does| do)|how much (?:is|are)|calculate|compute|work out|"
    r"figure out|tell me|give me|solve)?\s*",
)
TRAILER = re.compile(r"\s*(?:equal|equals|make|makes|come to|is)?\s*$")
MAX_EXPONENT = 1000
MAX_MAGNITUDE = 1e100


class MathError(ValueError):
    """Not something the calculator can answer."""


# Numbers ----------------------------------------------------------------------


def _read_number(words: list[str], i: int) -> tuple[float, int] | None:
    """A number in words or digits at ``words[i]``: "one thousand two hundred", "3.5"."""
    if i >= len(words):
        return None
    word = words[i]
    if re.fullmatch(r"\d[\d,]*(?:\.\d+)?|\.\d+", word):
        value, j = float(word.replace(",", "")), i + 1
    elif word in ("a", "an") and i + 1 < len(words) and words[i + 1] in ("hundred", *SCALES):
        value, j = 1.0, i + 1
    elif word in ONES or word in TENS or word == "zero":
        value, j = 0.0, i
    else:
        return None
    if j == i:
        total, current = 0.0, 0.0
        while j < len(words):
            word = words[j]
            if word in TENS:
                current += TENS[word]
            elif word in ONES:
                current += ONES[word]
            elif word == "hundred" and current:
                current *= 100
            elif word in SCALES and (current or total):
                total += (current or 1) * SCALES[word]
                current = 0.0
            elif word == "and" and j + 1 < len(words) and words[j + 1] in (*ONES, *TENS):
                pass  # "one hundred and five"
            else:
                break
            j += 1
        value = total + current
    else:
        # "1.5 million", "two hundred" after digits ("3 hundred")
        while j < len(words) and (words[j] in SCALES or words[j] == "hundred"):
            value *= SCALES.get(words[j], 100)
            j += 1
    # "three point one four"
    if words[j : j + 1] == ["point"] and j + 1 < len(words):
        digits = []
        k = j + 1
        while k < len(words) and (words[k] in ONES and ONES[words[k]] < 10 or words[k] == "oh"):
            digits.append(str(ONES.get(words[k], 0)))
            k += 1
        if digits:
            value += float("0." + "".join(digits))
            j = k
    return value, j


# Spoken expression -> Python expression ----------------------------------------

BINARY = [
    (("to", "the", "power", "of"), "**", "to the power of"),
    (("raised", "to", "the", "power", "of"), "**", "to the power of"),
    (("raised", "to", "the"), "**", "to the power of"),
    (("to", "the"), "**", "to the power of"),
    (("raised", "to"), "**", "to the power of"),
    (("multiplied", "by"), "*", "times"),
    (("divided", "by"), "/", "divided by"),
    (("percent", "of"), "%of", "percent of"),
    (("added", "to"), "+", "plus"),
    (("plus",), "+", "plus"),
    (("and",), "+", "plus"),
    (("minus",), "-", "minus"),
    (("less",), "-", "minus"),
    (("times",), "*", "times"),
    (("x",), "*", "times"),
    (("over",), "/", "divided by"),
    (("mod",), "%", "modulo"),
    (("modulo",), "%", "modulo"),
    (("+",), "+", "plus"),
    (("-",), "-", "minus"),
    (("*",), "*", "times"),
    (("/",), "/", "divided by"),
    (("^",), "**", "to the power of"),
]
PREFIX = [
    (("the", "square", "root", "of"), "sqrt", "the square root of"),
    (("square", "root", "of"), "sqrt", "the square root of"),
    (("the", "cube", "root", "of"), "cbrt", "the cube root of"),
    (("cube", "root", "of"), "cbrt", "the cube root of"),
    (("negative",), "neg", "negative"),
    (("minus",), "neg", "minus"),
]


@dataclass(slots=True, frozen=True)
class Expression:
    python: str  # for the evaluator
    spoken: str  # read back to the user


def _words(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    text = text.replace("×", " x ").replace("÷", " / ").replace("%", " percent ")
    text = re.sub(r"(?<=\d)\s*([+*/^])\s*(?=\d)", r" \1 ", text)
    text = re.sub(r"(?<=\d)x(?=\d)", " x ", text)
    text = re.sub(r"(?<![\w.])-(?=\d)", "minus ", text)
    text = re.sub(r"[?!]+|(?<!\d)\.(?!\d)", " ", text)
    return re.findall(r"\d+(?:st|nd|rd|th)\b|\d[\d,]*(?:\.\d+)?|\.\d+|[a-z']+|[+\-*/^()]", text)


def _match(words, i, table):
    for pattern, symbol, spoken in table:
        if tuple(words[i : i + len(pattern)]) == pattern:
            return symbol, spoken, i + len(pattern)
    return None


def parse_expression(text: str) -> Expression | None:
    """A Python expression for spoken arithmetic, or None if it is not arithmetic."""
    spoken_text = LEAD_IN.sub("", text.strip().lower().replace("’", "'"), count=1)
    words = _words(spoken_text)
    while words and words[-1] in ("equal", "equals", "make", "makes", "is", "please"):
        words.pop()
    python, spoken = [], []
    i, expect_value, operators = 0, True, 0
    while i < len(words):
        if expect_value:
            prefix = _match(words, i, PREFIX)
            if prefix:
                symbol, said, i = prefix
                python.append({"sqrt": "sqrt(", "cbrt": "cbrt(", "neg": "-("}[symbol])
                spoken.append(said)
                python_close = ")"
                number = _read_number(words, i)
                if number is None:
                    return None
                value, i = number
                python.append(f"{value!r}{python_close}")
                spoken.append(say_number(value))
                expect_value = False
                operators += symbol != "neg"
                continue
            number = _read_number(words, i)
            if number is None:
                return None
            value, i = number
            python.append(repr(value))
            spoken.append(say_number(value))
            expect_value = False
            # "five squared", "two cubed", "ten percent"
            if words[i : i + 1] == ["squared"]:
                python.append("**2")
                spoken.append("squared")
                operators += 1
                i += 1
            elif words[i : i + 1] == ["cubed"]:
                python.append("**3")
                spoken.append("cubed")
                operators += 1
                i += 1
            elif words[i : i + 1] == ["percent"] and words[i + 1 : i + 2] != ["of"]:
                python.append("/100")
                spoken.append("percent")
                i += 1
            continue
        operator_match = _match(words, i, BINARY)
        if operator_match is None:
            return None
        symbol, said, i = operator_match
        if symbol == "**" and i < len(words) and words[i] in ORDINAL_POWERS:
            # "two to the tenth"
            python.append(f"**{ORDINAL_POWERS[words[i]]}")
            spoken.append(f"to the power of {ORDINAL_POWERS[words[i]]}")
            operators += 1
            i += 1
            if words[i : i + 1] == ["power"]:
                i += 1
            continue
        if symbol == "**" and i < len(words):
            # "two to the 10th"
            ordinal = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", words[i])
            if ordinal:
                python.append(f"**{ordinal.group(1)}")
                spoken.append(f"to the power of {ordinal.group(1)}")
                operators += 1
                i += 1
                continue
        python.append("/100*" if symbol == "%of" else symbol)
        spoken.append(said)
        operators += 1
        expect_value = True
    if expect_value or not operators:
        return None
    return Expression(" ".join(python), " ".join(spoken))


# Evaluation ---------------------------------------------------------------------

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
FUNCTIONS = {
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "abs": abs,
    "round": round,
}


def evaluate(expression: str) -> float:
    """Evaluate +, -, *, /, %, //, ** and sqrt/cbrt/abs/round over numbers only."""
    text = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    # "15%" is a percentage; "10 % 3" is a remainder.
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(.])", r"(\1/100)", text)
    if len(text) > 200:
        raise MathError("That expression is too long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise MathError("That isn't an expression I can work out") from exc
    value = _walk(tree.body)
    if isinstance(value, complex) or math.isnan(value) or math.isinf(value):
        raise MathError("That doesn't have a real answer")
    if abs(value) > MAX_MAGNITUDE:
        raise MathError("That number is too big to say")
    return float(value)


def _walk(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _walk(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _walk(node.left), _walk(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT or (
                abs(left) > 1 and abs(right) * math.log10(abs(left)) > 100
            ):
                raise MathError("That number is too big to say")
            return float(left) ** right
        handler = OPERATORS.get(type(node.op))
        if handler is None:
            raise MathError("That isn't an expression I can work out")
        try:
            return handler(left, right)
        except ZeroDivisionError as exc:
            raise MathError("You can't divide by zero") from exc
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and len(node.args) == 1
        and not node.keywords
    ):
        argument = _walk(node.args[0])
        if node.func.id == "sqrt" and argument < 0:
            raise MathError("Negative numbers don't have a real square root")
        return FUNCTIONS[node.func.id](argument)
    raise MathError("That isn't an expression I can work out")


def say_number(value: float) -> str:
    """ "391", "14.29", "0.333", "1,250,000"."""
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9) and abs(value) < 1e15:
        return f"{int(round(value)):,}"
    magnitude = abs(value)
    if magnitude >= 1000:
        decimals = 1
    elif magnitude >= 1:
        decimals = 2
    else:
        decimals = max(2, 1 - math.floor(math.log10(magnitude)) + 1)
    text = f"{value:,.{decimals}f}".rstrip("0").rstrip(".")
    return text


# Units --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Unit:
    kind: str
    factor: float  # to the kind's base unit; for temperature, which scale
    singular: str
    plural: str


UNITS: dict[str, Unit] = {}


def _add(kind: str, factor: float, singular: str, plural: str, *aliases: str) -> None:
    unit = Unit(kind, factor, singular, plural)
    for name in (singular, plural, *aliases):
        UNITS.setdefault(name, unit)


_add("length", 0.001, "millimeter", "millimeters", "millimetre", "millimetres", "mm")
_add("length", 0.01, "centimeter", "centimeters", "centimetre", "centimetres", "cm")
_add("length", 1.0, "meter", "meters", "metre", "metres", "m")
_add("length", 1000.0, "kilometer", "kilometers", "kilometre", "kilometres", "km")
_add("length", 0.0254, "inch", "inches", "in")
_add("length", 0.3048, "foot", "feet", "ft")
_add("length", 0.9144, "yard", "yards", "yd")
_add("length", 1609.344, "mile", "miles", "mi")
_add("mass", 1e-6, "milligram", "milligrams", "mg")
_add("mass", 0.001, "gram", "grams", "g")
_add("mass", 1.0, "kilogram", "kilograms", "kilo", "kilos", "kg")
_add("mass", 0.028349523125, "ounce", "ounces", "oz")
_add("mass", 0.45359237, "pound", "pounds", "lb", "lbs")
_add("mass", 6.35029318, "stone", "stone", "stones")
_add("mass", 1000.0, "tonne", "tonnes", "metric ton", "metric tons")
_add("volume", 0.001, "milliliter", "milliliters", "millilitre", "millilitres", "ml")
_add("volume", 1.0, "liter", "liters", "litre", "litres", "l")
_add("volume", 0.00492892159375, "teaspoon", "teaspoons", "tsp")
_add("volume", 0.01478676478125, "tablespoon", "tablespoons", "tbsp")
_add("volume", 0.0295735295625, "fluid ounce", "fluid ounces", "fl oz")
_add("volume", 0.2365882365, "cup", "cups")
_add("volume", 0.473176473, "pint", "pints")
_add("volume", 0.946352946, "quart", "quarts")
_add("volume", 3.785411784, "gallon", "gallons")
_add("speed", 0.44704, "mile per hour", "miles per hour", "mph")
_add(
    "speed",
    1 / 3.6,
    "kilometer per hour",
    "kilometers per hour",
    "kilometre per hour",
    "kilometres per hour",
    "km/h",
    "kph",
)
_add("speed", 1.0, "meter per second", "meters per second", "m/s")
_add("speed", 0.514444, "knot", "knots")
_add("time", 1.0, "second", "seconds", "sec", "secs")
_add("time", 60.0, "minute", "minutes", "min", "mins")
_add("time", 3600.0, "hour", "hours", "hr", "hrs")
_add("time", 86400.0, "day", "days")
_add("time", 604800.0, "week", "weeks")
_add("temperature", 0.0, "degree Celsius", "degrees Celsius", "celsius", "centigrade", "c")
_add("temperature", 1.0, "degree Fahrenheit", "degrees Fahrenheit", "fahrenheit", "f")
_add("temperature", 2.0, "kelvin", "kelvin", "kelvins", "k")
UNITS = {name.lower(): unit for name, unit in UNITS.items()}

UNIT_NAMES = sorted(UNITS, key=len, reverse=True)
UNIT_PATTERN = "(?:" + "|".join(re.escape(name) for name in UNIT_NAMES) + ")"
NUMBER_WORDS = (
    r"[\d.,]+|(?:(?:a|an|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|point|and|\d+)\b\s*)+"
)
CONVERT = [
    re.compile(
        rf"^(?:convert|change|turn)?\s*(?P<value>{NUMBER_WORDS})\s*(?:degrees?\s+)?"
        rf"(?P<source>{UNIT_PATTERN})\s+(?:to|in|into|as)\s+(?:degrees?\s+)?"
        rf"(?P<target>{UNIT_PATTERN})$"
    ),
    re.compile(
        rf"^how many (?P<target>{UNIT_PATTERN}) (?:are |is )?(?:there )?"
        rf"(?:in |make up |make |per |equals? |equal to )?"
        rf"(?:(?P<value>{NUMBER_WORDS})\s*)?(?P<source>{UNIT_PATTERN})$"
    ),
]


@dataclass(slots=True, frozen=True)
class Conversion:
    value: float
    source: Unit
    target: Unit
    result: float

    def spoken(self) -> str:
        exact = math.isclose(self.result, round(self.result, 2), abs_tol=1e-9)
        about = "" if exact else "about "
        return (
            f"{say_number(self.value)} {_unit_name(self.source, self.value)} is "
            f"{about}{say_number(self.result)} {_unit_name(self.target, self.result)}."
        )


def _unit_name(unit: Unit, value: float) -> str:
    return unit.singular if math.isclose(abs(value), 1) else unit.plural


def convert(value: float, source: Unit, target: Unit) -> float:
    if source.kind != target.kind:
        raise MathError(f"You can't convert {source.plural} to {target.plural}")
    if source.kind == "temperature":
        celsius = {0.0: value, 1.0: (value - 32) * 5 / 9, 2.0: value - 273.15}[source.factor]
        return {0.0: celsius, 1.0: celsius * 9 / 5 + 32, 2.0: celsius + 273.15}[target.factor]
    return value * source.factor / target.factor


def parse_conversion(text: str) -> Conversion | None:
    spoken = LEAD_IN.sub("", text.strip().lower().replace("’", "'"), count=1)
    spoken = re.sub(r"[?!]+$|\.$", "", spoken).strip()
    spoken = re.sub(r"\s+", " ", spoken)
    for pattern in CONVERT:
        found = pattern.match(spoken)
        if not found:
            continue
        raw = (found.group("value") or "one").strip()
        if raw in ("a", "an"):
            raw = "one"
        number = _read_number(_words(raw), 0)
        if number is None or number[1] != len(_words(raw)):
            return None
        source, target = UNITS[found.group("source")], UNITS[found.group("target")]
        if source == target:
            return None
        try:
            result = convert(number[0], source, target)
        except MathError:
            return None
        return Conversion(number[0], source, target, result)
    return None
