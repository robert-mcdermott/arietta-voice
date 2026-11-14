from __future__ import annotations

import re
from datetime import date, timedelta

from arietta_voice.calendar_math import parse_date
from arietta_voice.timephrases import normalize
from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult
from arietta_voice.weather import (
    WeatherClient,
    WeatherError,
    describe,
    describe_rain,
    describe_weekend,
    local_today,
)

ABOUT_WEATHER = re.compile(
    r"\b(?:weather|forecast|rain|raining|rainy|umbrella|snow|snowing|temperature|"
    r"(?:how|is it) (?:hot|cold|warm|chilly)|sunny)\b"
)
NOT_WEATHER = re.compile(r"\b(?:convert|degrees? (?:celsius|fahrenheit)|timer|remind|alarm)\b")
DAY = re.compile(
    r"\b(?P<day>today|tonight|tomorrow|this (?:afternoon|evening|morning|weekend)|"
    r"the weekend|(?:on |this |next )?(?:monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday))\b"
)
PLACE = re.compile(r"\b(?:in|for|at) (?P<place>[a-z][a-z .,']{1,60}?)\s*$")
NOT_PLACES = {"the morning", "the afternoon", "the evening", "the weekend", "general"}


def weather_request(text: str) -> dict | None:
    """ "will it rain tomorrow in Paris" -> {"question": "rain", "day": "tomorrow", "place": "paris"}."""
    spoken = " ".join(re.sub(r"[^a-z0-9 ,.']+", " ", normalize(text)).split()).strip(" .,")
    if not ABOUT_WEATHER.search(spoken) or NOT_WEATHER.search(spoken):
        return None
    arguments: dict[str, str] = {}
    found = DAY.search(spoken)
    if found:
        arguments["day"] = found["day"]
        spoken = (spoken[: found.start()] + spoken[found.end() :]).strip()
        spoken = " ".join(spoken.split())
    place = PLACE.search(spoken)
    if place and place["place"].strip(" .,") not in NOT_PLACES:
        arguments["place"] = place["place"].strip(" .,")
    rain = re.search(r"\b(?:rain|raining|rainy|umbrella|snow|snowing|wet)\b", spoken)
    arguments["question"] = "rain" if rain else "forecast"
    return arguments


class WeatherTool(BaseTool):
    """Today's weather and the week's forecast, from Open-Meteo (opt-in)."""

    name = "weather"
    description = "the user asks about the weather, the temperature or rain, now or this week"
    parameters = {
        "type": "object",
        "properties": {
            "day": {"type": "string", "description": "such as today, tomorrow or Saturday"},
            "place": {
                "type": "string",
                "description": "a town or city, only if the user named one",
                "maxLength": 80,
            },
            "question": {"type": "string", "enum": ["forecast", "rain"]},
        },
    }
    aliases = {"location": "place", "city": "place", "date": "day", "when": "day"}
    speaks_for_itself = True
    timeout_s = 15.0

    def __init__(self, context=None, client=None, clock=local_today) -> None:
        super().__init__(context)
        settings = context.config.weather
        self.home, self.units = settings.location, settings.units
        self.client = client if client is not None else WeatherClient()
        self.clock = clock

    def match(self, text):
        return weather_request(text)

    def run(self, arguments):
        today = self.clock()
        named = str(arguments.get("place") or "").strip()
        try:
            place = self.client.place(named or self.home)
            if place is None:
                where = named or self.home
                return _say(f"Sorry, I couldn't find {where}.")
            forecast = self.client.forecast(place, self.units)
        except WeatherError:
            return _say("Sorry, I couldn't reach the weather service.")
        day_text = str(arguments.get("day") or "today").strip().lower()
        say_place = bool(named)
        if named and forecast.today() is not None:
            today = forecast.today()  # Tomorrow in Tokyo is Tokyo's tomorrow.
        if "weekend" in day_text:
            return _say(describe_weekend(forecast, today, say_place=say_place))
        day = _resolve_day(day_text, today)
        if day is None:
            return _say("I only have the forecast for today and the next six days.")
        if arguments.get("question") == "rain":
            return _say(describe_rain(forecast, day, today, umbrella=True))
        return _say(describe(forecast, day, today, say_place=say_place))


def _resolve_day(text: str, today: date) -> date | None:
    text = re.sub(r"^(?:on|this|next)\s+", "", text)
    if text in ("today", "tonight", "now", "afternoon", "evening", "morning"):
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    if text in names:  # this coming one, or today
        return today + timedelta(days=(names.index(text) - today.weekday()) % 7)
    found = parse_date(text, today)
    if found is None or not 0 <= (found.day - today).days <= 6:
        return None
    return found.day


def _say(answer: str) -> ToolResult:
    return ToolResult(content=answer, speak=answer)


def create_tool(context: ToolContext) -> WeatherTool | None:
    if not context.config.weather.location:
        return None  # Opt-in: set [weather].location to use it.
    return WeatherTool(context)
