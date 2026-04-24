"""
Lookup Provider Tests - Antique Telephone AI Operator

Tests for the LookupProvider interface and the GooglePlacesProvider implementation.
All HTTP calls are mocked so no real API key or network is required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.base import LookupProvider, LookupResult
from providers.lookup.google_places_provider import GooglePlacesProvider
from utils.config_manager import ConfigManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_with_key(api_key="test_api_key", home_lat=None, home_lon=None, radius=50000):
    config = ConfigManager()
    config.set("lookup.google_api_key", api_key)
    if home_lat is not None:
        config.set("lookup.home_lat", home_lat)
    if home_lon is not None:
        config.set("lookup.home_lon", home_lon)
    config.set("lookup.radius_meters", radius)
    return config


def _places_response(places: list) -> dict:
    """Build a fake Places API JSON response."""
    return {"places": places}


def _place(name="Joe's Pizza", address="123 Main St, Columbus, OH", phone="+1 614-555-1234", rating=4.5):
    return {
        "displayName": {"text": name},
        "formattedAddress": address,
        "nationalPhoneNumber": phone,
        "rating": rating,
    }


# ---------------------------------------------------------------------------
# LookupProvider interface
# ---------------------------------------------------------------------------

class TestLookupProviderInterface:
    """Verify the ABC contract is enforced."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            LookupProvider()  # type: ignore

    def test_concrete_subclass_must_implement_search(self):
        class Incomplete(LookupProvider):
            @property
            def is_available(self):
                return True
            # search not implemented

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore

    def test_lookup_result_is_dataclass(self):
        r = LookupResult(name="Foo", address="1 Main", phone_number="6140000001", confidence=0.9)
        assert r.name == "Foo"
        assert r.phone_number == "6140000001"
        assert r.confidence == 0.9


# ---------------------------------------------------------------------------
# GooglePlacesProvider — unit tests (HTTP mocked)
# ---------------------------------------------------------------------------

class TestGooglePlacesProvider:

    def test_is_available_with_key_and_aiohttp(self):
        config = _config_with_key("my_key")
        provider = GooglePlacesProvider(config)
        # aiohttp is available in test environment
        assert provider.is_available is True

    def test_is_unavailable_without_key(self):
        config = ConfigManager()
        config.set("lookup.google_api_key", "")
        provider = GooglePlacesProvider(config)
        assert provider.is_available is False

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_unavailable(self):
        config = ConfigManager()
        config.set("lookup.google_api_key", "")
        provider = GooglePlacesProvider(config)
        results = await provider.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_single_result(self):
        config = _config_with_key()
        provider = GooglePlacesProvider(config)

        fake_resp = _places_response([_place()])
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=fake_resp)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_response)

        with patch("providers.lookup.google_places_provider.aiohttp.ClientSession", return_value=mock_session):
            results = await provider.search("Joe's Pizza")

        assert len(results) == 1
        assert results[0].name == "Joe's Pizza"
        assert results[0].phone_number == "6145551234"  # normalised
        assert results[0].confidence == pytest.approx(0.9, rel=1e-3)

    @pytest.mark.asyncio
    async def test_search_multiple_results(self):
        config = _config_with_key()
        provider = GooglePlacesProvider(config)

        fake_resp = _places_response([
            _place("Pizza A", phone="+1 614-111-1111", rating=5.0),
            _place("Pizza B", phone="+1 614-222-2222", rating=3.0),
        ])
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=fake_resp)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_response)

        with patch("providers.lookup.google_places_provider.aiohttp.ClientSession", return_value=mock_session):
            results = await provider.search("pizza")

        assert len(results) == 2
        assert results[0].name == "Pizza A"
        assert results[1].name == "Pizza B"

    @pytest.mark.asyncio
    async def test_search_filters_results_without_phone(self):
        config = _config_with_key()
        provider = GooglePlacesProvider(config)

        no_phone = {
            "displayName": {"text": "No Phone Place"},
            "formattedAddress": "1 Oak St",
            # nationalPhoneNumber absent
        }
        fake_resp = _places_response([no_phone, _place()])
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=fake_resp)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_response)

        with patch("providers.lookup.google_places_provider.aiohttp.ClientSession", return_value=mock_session):
            results = await provider.search("test")

        assert len(results) == 1
        assert results[0].name == "Joe's Pizza"

    @pytest.mark.asyncio
    async def test_search_includes_location_bias_when_configured(self):
        config = _config_with_key(home_lat=39.96, home_lon=-82.99, radius=20000)
        provider = GooglePlacesProvider(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"places": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_response)

        with patch("providers.lookup.google_places_provider.aiohttp.ClientSession", return_value=mock_session):
            await provider.search("pizza", lat=39.96, lon=-82.99)

        _, call_kwargs = mock_session.post.call_args
        payload = call_kwargs["json"]
        assert "locationBias" in payload
        circle = payload["locationBias"]["circle"]
        assert circle["center"]["latitude"] == pytest.approx(39.96)
        assert circle["center"]["longitude"] == pytest.approx(-82.99)
        assert circle["radius"] == pytest.approx(20000.0)

    @pytest.mark.asyncio
    async def test_search_no_location_bias_when_not_configured(self):
        config = _config_with_key()
        provider = GooglePlacesProvider(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"places": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_response)

        with patch("providers.lookup.google_places_provider.aiohttp.ClientSession", return_value=mock_session):
            await provider.search("pizza")  # no lat/lon

        _, call_kwargs = mock_session.post.call_args
        payload = call_kwargs["json"]
        assert "locationBias" not in payload

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_api_error(self):
        config = _config_with_key()
        provider = GooglePlacesProvider(config)

        mock_response = AsyncMock()
        mock_response.status = 403
        mock_response.text = AsyncMock(return_value="Forbidden")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_response)

        with patch("providers.lookup.google_places_provider.aiohttp.ClientSession", return_value=mock_session):
            results = await provider.search("anything")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_network_exception(self):
        config = _config_with_key()
        provider = GooglePlacesProvider(config)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(side_effect=Exception("Network error"))

        with patch("providers.lookup.google_places_provider.aiohttp.ClientSession", return_value=mock_session):
            results = await provider.search("anything")

        assert results == []


# ---------------------------------------------------------------------------
# Phone normalisation
# ---------------------------------------------------------------------------

class TestNormalisePhone:

    def test_strips_formatting(self):
        assert GooglePlacesProvider._normalise_phone("+1 614-555-1234") == "6145551234"

    def test_drops_leading_country_code(self):
        assert GooglePlacesProvider._normalise_phone("16145551234") == "6145551234"

    def test_local_number_unchanged(self):
        assert GooglePlacesProvider._normalise_phone("6145551234") == "6145551234"

    def test_seven_digit_number(self):
        assert GooglePlacesProvider._normalise_phone("555-1234") == "5551234"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
