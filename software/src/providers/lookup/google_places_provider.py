"""
Google Places Lookup Provider - Antique Telephone AI Operator

Implements LookupProvider using the Google Places API (New) searchText endpoint.
Returns ranked business matches for a text query, optionally biased toward a
home location for proximity-aware results.

Config keys used:
  lookup.google_api_key  — GOOGLE_PLACES_API_KEY env var also accepted
  lookup.home_lat        — latitude for location bias (optional)
  lookup.home_lon        — longitude for location bias (optional)
  lookup.radius_meters   — search radius in metres, default 50000
"""

import asyncio
import logging
from typing import List, Optional

from providers.base import LookupProvider, LookupResult
from utils.config_manager import ConfigManager

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class GooglePlacesProvider(LookupProvider):
    """Looks up businesses via the Google Places API (New) Text Search endpoint."""

    _ENDPOINT = "https://places.googleapis.com/v1/places:searchText"
    _FIELD_MASK = "places.displayName,places.formattedAddress,places.nationalPhoneNumber,places.rating"

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        self._api_key: str = self.config.get("lookup.google_api_key") or ""
        self._project_id: str = self.config.get("lookup.gcp_project_id") or ""

    @property
    def is_available(self) -> bool:
        return HAS_AIOHTTP and bool(self._api_key)

    async def search(
        self,
        query: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> List[LookupResult]:
        """Search Google Places for *query* and return up to 5 results."""
        if not self.is_available:
            self.logger.warning("GooglePlacesProvider unavailable (missing aiohttp or API key)")
            return []

        radius = int(self.config.get("lookup.radius_meters", 50000))

        payload: dict = {"textQuery": query, "maxResultCount": 5}

        if lat is not None and lon is not None:
            payload["locationBias"] = {
                "circle": {
                    "center": {"latitude": lat, "longitude": lon},
                    "radius": float(radius),
                }
            }

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": self._FIELD_MASK,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._ENDPOINT,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        self.logger.error(f"Places API error {resp.status}: {body}")
                        return []
                    data = await resp.json()
        except Exception as exc:
            self.logger.error(f"Places API request failed: {exc}")
            return []

        results: List[LookupResult] = []
        for place in data.get("places", []):
            phone = place.get("nationalPhoneNumber", "")
            if not phone:
                continue  # skip listings without a phone number
            name = place.get("displayName", {}).get("text", "Unknown")
            address = place.get("formattedAddress", "")
            rating = place.get("rating", 0.0)
            # Map 1–5 star rating to 0–1 confidence; unrated places get 0.5
            confidence = (rating / 5.0) if rating else 0.5
            results.append(LookupResult(
                name=name,
                address=address,
                phone_number=self._normalise_phone(phone),
                confidence=confidence,
                source="business",
            ))

        self.logger.info(f"Places search '{query}' returned {len(results)} results")
        return results

    @staticmethod
    def _normalise_phone(raw: str) -> str:
        """Strip formatting so the number is plain digits (e.g. '6145989581')."""
        import re
        digits = re.sub(r"\D", "", raw)
        # Drop leading country code '1' for North American numbers
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return digits
