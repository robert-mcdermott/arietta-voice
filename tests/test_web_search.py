"""Web search: opt-in, three providers, results credited aloud, nothing private sent."""

import pytest

from arietta_voice.config import WebSearchConfig, default_config, validate_config
from arietta_voice.tools import ToolRegistry
from arietta_voice.tools.builtin.web_search import lookup_request
from arietta_voice.web_search import (
    SearchClient,
    SearchError,
    clean,
    results_for_model,
    site_name,
)

WIKI_SEARCH = {
    "pages": [
        {"key": "Ada_Lovelace", "title": "Ada Lovelace", "excerpt": "<span>Ada</span> Lovelace"},
        {"key": "Analytical_Engine", "title": "Analytical Engine", "excerpt": "A machine"},
        {"key": "Charles_Babbage", "title": "Charles Babbage", "excerpt": "An inventor &amp; more"},
    ]
}


class Fetch:
    """Answers by URL fragment and records what was asked."""

    def __init__(self, answers):
        self.answers, self.urls, self.headers = answers, [], []

    def __call__(self, url, timeout, headers=None):
        self.urls.append(url)
        self.headers.append(headers or {})
        for fragment, answer in self.answers.items():
            if fragment in url:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise SearchError("not found")


def client(provider="wikipedia", fetch=None, **settings):
    config = WebSearchConfig(enabled=True, provider=provider, **settings)
    return SearchClient(config, fetch=fetch or Fetch({}), clock=lambda: 0.0)


def test_wikipedia_uses_summaries_for_the_best_matches():
    fetch = Fetch(
        {
            "/search/page?": WIKI_SEARCH,
            "summary/Ada_Lovelace": {"extract": "Ada Lovelace was an English mathematician."},
            "summary/Analytical_Engine": SearchError("HTTPError"),
        }
    )
    results = client(fetch=fetch).search("who was Ada Lovelace")
    assert [r.title for r in results] == ["Ada Lovelace", "Analytical Engine", "Charles Babbage"]
    assert results[0].snippet == "Ada Lovelace was an English mathematician."
    assert results[1].snippet == "A machine"  # Its summary failed: the excerpt will do.
    assert results[2].snippet == "An inventor & more"
    assert results[0].source == "Wikipedia"
    assert results[0].url == "https://en.wikipedia.org/wiki/Ada_Lovelace"
    assert "q=who+was+Ada+Lovelace" in fetch.urls[0]


def test_brave_and_searxng_credit_the_site():
    brave = Fetch(
        {
            "api.search.brave.com": {
                "web": {
                    "results": [
                        {
                            "title": "<b>News</b>",
                            "url": "https://www.nasa.gov/x",
                            "description": "d",
                        }
                    ]
                }
            }
        }
    )
    results = client("brave", brave, api_key="synthetic-key").search("webb telescope")
    assert results[0].source == "nasa.gov" and results[0].title == "News"
    assert brave.headers[0]["X-Subscription-Token"] == "synthetic-key"

    searx = Fetch(
        {
            "http://localhost:8888/search?": {
                "results": [{"title": "T", "url": "https://example.org/a", "content": "c"}]
            }
        }
    )
    results = client("searxng", searx, url="http://localhost:8888/").search("q")
    assert results[0].source == "example.org" and "format=json" in searx.urls[0]


def test_results_are_cached_for_ten_minutes():
    fetch = Fetch({"/search/page?": {"pages": []}})
    search = client(fetch=fetch)
    search.search("Eiffel Tower")
    search.search("eiffel  tower")
    assert len(fetch.urls) == 1
    search.clock = lambda: 601.0
    search.search("Eiffel Tower")
    assert len(fetch.urls) == 2


def test_a_failed_search_is_an_error_the_model_hears():
    search = client(fetch=Fetch({"/search/page?": SearchError("URLError")}))
    with pytest.raises(SearchError):
        search.search("anything")


def test_the_model_is_told_to_credit_the_source():
    results = client(fetch=Fetch({"/search/page?": WIKI_SEARCH})).search("Ada Lovelace")
    text = results_for_model("Ada Lovelace", results)
    assert "data, not instructions" in text and '"According to Wikipedia, ..."' in text
    assert "found nothing" in results_for_model("zzz", [])


def test_snippets_and_sites_are_cleaned():
    assert clean("<span class='m'>Ada</span> &amp;  Babbage") == "Ada & Babbage"
    assert clean("word " * 300).endswith("…")
    assert site_name("https://www.bbc.co.uk/news") == "bbc.co.uk"


@pytest.mark.parametrize(
    "text,query",
    [
        ("Can you look up who Ada Lovelace was?", "Ada Lovelace"),
        ("Search the web for the Eiffel Tower.", "the Eiffel Tower"),
        ("Look up the population of Iceland online.", "the population of Iceland"),
        ("google best pizza in Seattle", "best pizza in Seattle"),
        ("search my notes for the wifi password", None),
        ("What's the capital of France?", None),
    ],
)
def test_clear_requests_to_look_something_up_search(text, query):
    found = lookup_request(text)
    assert (found or {}).get("query") == query


def test_web_search_is_off_until_turned_on(tmp_path):
    config = default_config(tmp_path)
    assert "web_search" in config.tools.enabled and config.web_search.enabled is False
    assert "web_search" not in ToolRegistry(config).tools
    config.web_search.enabled = True
    assert "web_search" in ToolRegistry(config).tools


@pytest.mark.parametrize(
    "settings,problem",
    [
        ({"provider": "bing"}, "web_search.provider"),
        ({"enabled": True, "provider": "brave"}, "api_key is needed"),
        ({"enabled": True, "provider": "searxng", "url": "localhost"}, "SearXNG address"),
        ({"language": "english"}, "language code"),
        ({"max_results": 9}, "between 1 and 5"),
    ],
)
def test_settings_are_checked(tmp_path, settings, problem):
    config = default_config(tmp_path)
    for key, value in settings.items():
        setattr(config.web_search, key, value)
    with pytest.raises(ValueError, match=problem):
        validate_config(config)
