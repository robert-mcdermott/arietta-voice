from datetime import date, timedelta

import pytest

from arietta_voice.config import default_config
from arietta_voice.tools import ToolContext, ToolRegistry
from arietta_voice.tools.builtin import weather as weather_tool
from arietta_voice.tools.builtin.weather import WeatherTool, weather_request
from arietta_voice.tools.registry import run_tool
from arietta_voice.weather import WeatherClient, WeatherError

TODAY = date(2026, 11, 14)  # A Saturday.
GEOCODE = {
    "results": [
        {"name": "Portland", "latitude": 45.52, "longitude": -122.68, "admin1": "Oregon",
         "country": "United States", "country_code": "US"},
        {"name": "Portland", "latitude": 43.66, "longitude": -70.26, "admin1": "Maine",
         "country": "United States", "country_code": "US"},
    ]
}  # fmt: skip


def forecast_json(codes=(61, 2, 0, 3, 80, 71, 1), rain=(80, 20, 0, 5, 60, 40, 10)):
    days = [(TODAY + timedelta(days=i)).isoformat() for i in range(7)]
    return {
        "current": {
            "temperature_2m": 51.4,
            "apparent_temperature": 45.2,
            "weather_code": 61,
            "wind_speed_10m": 12.0,
        },
        "daily": {
            "time": days,
            "weather_code": list(codes),
            "temperature_2m_max": [58.2, 60.0, 62.5, 55.0, 50.0, 33.0, 57.0],
            "temperature_2m_min": [47.0, 45.4, 44.0, 40.0, 41.0, 28.0, 42.0],
            "precipitation_probability_max": list(rain),
        },
    }


class FakeService:
    def __init__(self, forecast=None, fail=False):
        self.calls = []
        self.forecast = forecast or forecast_json()
        self.fail = fail

    def __call__(self, url, timeout):
        self.calls.append(url)
        if self.fail:
            raise WeatherError("TimeoutError")
        if "geocoding" in url:
            if "nowhereville" in url:
                return {}
            return GEOCODE
        return self.forecast


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What's the weather like?", {"question": "forecast"}),
        ("what's the weather tomorrow", {"day": "tomorrow", "question": "forecast"}),
        ("What's the weather in Paris tomorrow?", {"day": "tomorrow", "place": "paris",
                                                   "question": "forecast"}),
        ("will it rain tomorrow", {"day": "tomorrow", "question": "rain"}),
        ("Do I need an umbrella today?", {"day": "today", "question": "rain"}),
        ("is it going to snow on friday", {"day": "on friday", "question": "rain"}),
        ("what's the forecast for this weekend", {"day": "this weekend", "question": "forecast"}),
        ("how cold is it outside", {"question": "forecast"}),
        ("what's the temperature in Portland, Maine", {"place": "portland, maine",
                                                       "question": "forecast"}),
    ],
)  # fmt: skip
def test_weather_requests_are_recognised(text, expected):
    assert weather_request(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "what's 70 degrees fahrenheit in celsius",
        "set a timer for ten minutes",
        "remind me to bring an umbrella at 8",
        "what time is it",
        "tell me about the history of rainbows",
    ],
)
def test_other_requests_are_not_weather(text):
    assert weather_request(text) is None


def test_places_are_looked_up_once_and_qualified_by_region():
    service = FakeService()
    client = WeatherClient(fetch=service)
    assert client.place("Portland").region == "Oregon"
    assert client.place("Portland, Maine").region == "Maine"
    assert client.place("portland") is client.place("Portland")
    assert client.place("Nowhereville") is None
    assert sum("geocoding" in url for url in service.calls) == 3


def test_forecasts_are_cached_for_ten_minutes():
    service, now = FakeService(), [0.0]
    client = WeatherClient(fetch=service, clock=lambda: now[0])
    place = client.place("Portland")
    client.forecast(place)
    client.forecast(place)
    now[0] += 601
    client.forecast(place)
    forecasts = [url for url in service.calls if "forecast" in url]
    assert len(forecasts) == 2
    assert "temperature_unit=fahrenheit" in forecasts[0]
    client.forecast(place, "metric")
    assert "temperature_unit=celsius" in service.calls[-1]


def test_an_unexpected_forecast_is_an_error():
    client = WeatherClient(fetch=FakeService(forecast={"current": {}}))
    with pytest.raises(WeatherError):
        client.forecast(client.place("Portland"))


@pytest.fixture
def make_tool(tmp_path):
    def build(location="Portland", service=None):
        config = default_config(tmp_path)
        config.weather.location = location
        client = WeatherClient(fetch=service or FakeService())
        return WeatherTool(ToolContext(config), client=client, clock=lambda: TODAY)

    return build


@pytest.mark.parametrize(
    "arguments,speak",
    [
        (
            {},
            "Right now it's 51 degrees with light rain, feeling like 45. Today, a high of 58 "
            "and a low of 47, with an 80 percent chance of rain.",
        ),
        (
            {"day": "tomorrow"},
            "Tomorrow: partly cloudy skies, a high of 60 and a low of 45, with a 20 percent "
            "chance of rain.",
        ),
        (
            {"day": "monday"},
            "On Monday: clear skies, a high of 62 and a low of 44, and no rain expected.",
        ),
        (
            {"day": "thursday", "place": "Portland, Maine"},
            "On Thursday in Portland: light snow, a high of 33 and a low of 28, with a 40 percent "
            "chance of snow.",
        ),
        ({"day": "today", "question": "rain"},
         "Yes, light rain today, with an 80 percent chance of rain. You'll want an umbrella."),
        ({"day": "tomorrow", "question": "rain"},
         "Probably not: only a 20 percent chance of rain tomorrow."),
        ({"day": "wednesday", "question": "rain"},
         "Yes, light showers on Wednesday, with a 60 percent chance of rain. You'll want an "
         "umbrella."),
        ({"day": "this weekend"},
         "This weekend: Saturday, light rain with a high of 58, with an 80 percent chance of "
         "rain; Sunday, partly cloudy skies with a high of 60, with a 20 percent chance of "
         "rain."),
        ({"day": "in two weeks"}, "I only have the forecast for today and the next six days."),
    ],
)  # fmt: skip
def test_spoken_forecasts(make_tool, arguments, speak):
    assert run_tool(make_tool(), arguments).speak == speak


def test_unknown_places_and_an_unreachable_service_are_said_plainly(make_tool):
    assert run_tool(make_tool(), {"place": "Nowhereville"}).speak == (
        "Sorry, I couldn't find Nowhereville."
    )
    offline = make_tool(service=FakeService(fail=True))
    assert run_tool(offline, {}).speak == "Sorry, I couldn't reach the weather service."


def test_weather_is_opt_in(tmp_path, monkeypatch):
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    config.tools.enabled = (*config.tools.enabled, "weather")
    registry = ToolRegistry(config)
    assert "weather" not in registry.tools  # No location set.
    config.weather.location = "Portland"
    monkeypatch.setattr(weather_tool, "WeatherClient", lambda: WeatherClient(fetch=FakeService()))
    registry = ToolRegistry(config)
    registry.tools["weather"].clock = lambda: TODAY
    hit = registry.fast_path("Will it rain tomorrow?")
    assert hit.name == "weather"
    assert hit.result.speak == "Probably not: only a 20 percent chance of rain tomorrow."
    assert registry.fast_path("what's 70 degrees fahrenheit in celsius").name == "calculate"
    assert "weather" not in default_config(tmp_path).tools.enabled


def test_a_distant_place_uses_its_own_date(make_tool, monkeypatch):
    from arietta_voice import weather

    data = forecast_json()
    data["utc_offset_seconds"] = 9 * 3600  # Tokyo
    tool = make_tool(service=FakeService(forecast=data))
    # Late on the 13th here is already the 14th in Tokyo.
    real_datetime = weather.datetime

    class Clock(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 11, 13, 20, 0, tzinfo=tz)

    monkeypatch.setattr(weather, "datetime", Clock)
    tool.clock = lambda: date(2026, 11, 13)
    tomorrow = run_tool(tool, {"day": "tomorrow", "place": "Tokyo"}).speak
    assert tomorrow.startswith("Tomorrow in Portland: partly cloudy skies")


def test_odd_service_replies_are_reported_as_unreachable(make_tool):
    class Broken(FakeService):
        def __call__(self, url, timeout):
            if "geocoding" in url:
                return {"results": [{"name": "Nowhere"}]}  # No coordinates.
            return super().__call__(url, timeout)

    assert run_tool(make_tool(service=Broken()), {}).speak == (
        "Sorry, I couldn't reach the weather service."
    )
