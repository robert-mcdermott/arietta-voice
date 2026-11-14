"""Weather from Open-Meteo, said briefly.

Open-Meteo needs no account or key. What leaves the machine: the place name
(to find its coordinates, once per place) and those coordinates (for the
forecast). Results are cached for ten minutes, so repeated questions do not
repeat requests.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "arietta-voice"
CACHE_S = 600

# WMO weather interpretation codes, as Open-Meteo reports them.
CONDITIONS = {
    0: "clear skies",
    1: "mostly clear skies",
    2: "partly cloudy skies",
    3: "overcast skies",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "light snow showers",
    86: "snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with heavy hail",
}
WET = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}
SNOWY = {71, 73, 75, 77, 85, 86}


class WeatherError(Exception):
    """The weather service could not answer."""


@dataclass(slots=True, frozen=True)
class Place:
    name: str
    latitude: float
    longitude: float
    region: str = ""
    country: str = ""


@dataclass(slots=True, frozen=True)
class Day:
    day: date
    code: int
    high: float
    low: float
    rain_chance: int | None

    @property
    def conditions(self) -> str:
        return CONDITIONS.get(self.code, "mixed weather")


@dataclass(slots=True, frozen=True)
class Forecast:
    place: Place
    temperature: float
    feels_like: float
    code: int
    wind: float
    days: list[Day]
    units: str  # "imperial" or "metric"
    utc_offset_s: int | None = None  # the place's offset; its days are its own dates

    def today(self) -> date | None:
        """The date where the place is, or None if the service did not say."""
        if self.utc_offset_s is None:
            return None
        return (datetime.now(timezone.utc) + timedelta(seconds=self.utc_offset_s)).date()

    def on(self, day: date) -> Day | None:
        return next((item for item in self.days if item.day == day), None)


def _fetch_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise WeatherError(type(exc).__name__) from exc


class WeatherClient:
    def __init__(
        self,
        *,
        fetch: Callable[[str, float], dict] = _fetch_json,
        timeout_s: float = 6.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fetch, self.timeout_s, self.clock = fetch, timeout_s, clock
        self._places: dict[str, Place | None] = {}
        self._forecasts: dict[tuple, tuple[float, Forecast]] = {}

    def place(self, name: str) -> Place | None:
        """Coordinates for a place name ("Seattle", "Paris, France"), or None if unknown."""
        key = " ".join(name.lower().split())
        if key in self._places:
            return self._places[key]
        city, _, qualifier = key.partition(",")
        query = urllib.parse.urlencode(
            {"name": city.strip(), "count": 10, "language": "en", "format": "json"}
        )
        results = self.fetch(f"{GEOCODE_URL}?{query}", self.timeout_s).get("results") or []
        qualifier = qualifier.strip()
        if qualifier:
            # "Portland, Maine": prefer a result whose region or country matches.
            matching = [
                item
                for item in results
                if qualifier in str(item.get("admin1", "")).lower()
                or qualifier in str(item.get("country", "")).lower()
                or qualifier == str(item.get("country_code", "")).lower()
            ]
            results = matching or results
        found = None
        if results:
            best = results[0]
            try:
                found = Place(
                    name=str(best.get("name", name)),
                    latitude=float(best["latitude"]),
                    longitude=float(best["longitude"]),
                    region=str(best.get("admin1", "")),
                    country=str(best.get("country", "")),
                )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise WeatherError("Unexpected place") from exc
        self._places[key] = found
        return found

    def forecast(self, place: Place, units: str = "imperial") -> Forecast:
        key = (round(place.latitude, 3), round(place.longitude, 3), units)
        cached = self._forecasts.get(key)
        if cached and self.clock() - cached[0] < CACHE_S:
            return cached[1]
        imperial = units == "imperial"
        query = urllib.parse.urlencode(
            {
                "latitude": place.latitude,
                "longitude": place.longitude,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 7,
                "temperature_unit": "fahrenheit" if imperial else "celsius",
                "wind_speed_unit": "mph" if imperial else "kmh",
            }
        )
        data = self.fetch(f"{FORECAST_URL}?{query}", self.timeout_s)
        try:
            current, daily = data["current"], data["daily"]
            days = [
                Day(
                    day=date.fromisoformat(daily["time"][i]),
                    code=int(daily["weather_code"][i]),
                    high=float(daily["temperature_2m_max"][i]),
                    low=float(daily["temperature_2m_min"][i]),
                    rain_chance=(
                        None
                        if daily.get("precipitation_probability_max", [None] * 7)[i] is None
                        else int(daily["precipitation_probability_max"][i])
                    ),
                )
                for i in range(len(daily["time"]))
            ]
            forecast = Forecast(
                place=place,
                temperature=float(current["temperature_2m"]),
                feels_like=float(current["apparent_temperature"]),
                code=int(current["weather_code"]),
                wind=float(current["wind_speed_10m"]),
                days=days,
                units=units,
                utc_offset_s=(
                    int(data["utc_offset_seconds"])
                    if isinstance(data.get("utc_offset_seconds"), (int, float))
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise WeatherError("Unexpected forecast") from exc
        self._forecasts[key] = (self.clock(), forecast)
        return forecast


def _a(number: int) -> str:
    """ "a 20", "an 80", "an 11"."""
    return "an" if str(number).startswith("8") or number in (11, 18) else "a"


def _degrees(value: float) -> str:
    return f"{round(value)}"


def _rain(day: Day) -> str:
    if day.rain_chance is None:
        return ""
    kind = "snow" if day.code in SNOWY else "rain"
    if day.rain_chance < 10 and day.code not in WET | SNOWY:
        return f", and no {kind} expected"
    return f", with {_a(day.rain_chance)} {day.rain_chance} percent chance of {kind}"


def describe(forecast: Forecast, day: date, today: date, *, say_place: bool) -> str:
    """ "Tomorrow in Seattle: light rain, a high of 58 and a low of 47, ..." """
    where = f" in {forecast.place.name}" if say_place else ""
    item = forecast.on(day)
    if item is None:
        return "I only have the forecast for the next week."
    if day == today:
        now = (
            f"Right now{where} it's {_degrees(forecast.temperature)} degrees with "
            f"{CONDITIONS.get(forecast.code, 'mixed weather')}"
        )
        if abs(forecast.feels_like - forecast.temperature) >= 5:
            now += f", feeling like {_degrees(forecast.feels_like)}"
        return (
            f"{now}. Today, a high of {_degrees(item.high)} and a low of "
            f"{_degrees(item.low)}{_rain(item)}."
        )
    label = "Tomorrow" if (day - today).days == 1 else f"On {day.strftime('%A')}"
    return (
        f"{label}{where}: {item.conditions}, a high of {_degrees(item.high)} and a low of "
        f"{_degrees(item.low)}{_rain(item)}."
    )


def describe_rain(forecast: Forecast, day: date, today: date, *, umbrella: bool) -> str:
    item = forecast.on(day)
    if item is None:
        return "I only have the forecast for the next week."
    when = (
        "today"
        if day == today
        else "tomorrow"
        if (day - today).days == 1
        else (f"on {day.strftime('%A')}")
    )
    chance = item.rain_chance if item.rain_chance is not None else (60 if item.code in WET else 0)
    if item.code in WET or chance >= 50:
        advice = " You'll want an umbrella." if umbrella else ""
        return (
            f"Yes, {item.conditions} {when}, with {_a(chance)} {chance} percent chance of "
            f"rain.{advice}"
        )
    if chance >= 25:
        advice = " An umbrella wouldn't hurt." if umbrella else ""
        return f"Maybe: there's {_a(chance)} {chance} percent chance of rain {when}.{advice}"
    return f"Probably not: only {_a(chance)} {chance} percent chance of rain {when}."


def describe_weekend(forecast: Forecast, today: date, *, say_place: bool) -> str:
    saturday = today.toordinal() + (5 - today.weekday()) % 7
    days = [forecast.on(date.fromordinal(saturday + offset)) for offset in (0, 1)]
    days = [item for item in days if item is not None]
    if not days:
        return "I don't have the forecast for the weekend yet."
    where = f" in {forecast.place.name}" if say_place else ""
    parts = [
        f"{item.day.strftime('%A')}, {item.conditions} with a high of {_degrees(item.high)}"
        f"{_rain(item)}"
        for item in days
    ]
    return f"This weekend{where}: " + "; ".join(parts) + "."


def local_today() -> date:
    return datetime.now().astimezone().date()
