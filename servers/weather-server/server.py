import json
import os
import urllib.error
import urllib.parse
import urllib.request

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather Server")

WTTR_URL = "https://wttr.in"


def _fetch_json(url: str) -> dict | None:
    """Fetch JSON from a URL, returning None on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MCP-Weather-Server/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: City name (e.g., "Tokyo", "New York", "London")
    """
    url = f"{WTTR_URL}/{urllib.parse.quote(city)}?format=j1"
    data = _fetch_json(url)

    if data is None:
        return json.dumps({"error": f"Could not fetch weather data for '{city}'"})

    try:
        current = data["current_condition"][0]
        result = {
            "city": city,
            "temperature": f"{current['temp_C']}℃",        # ℃
            "condition": current["weatherDesc"][0]["value"],
            "humidity": f"{current['humidity']}%",
            "wind": f"{current['windspeedKmph']} km/h",
            "feels_like": f"{current['FeelsLikeC']}℃",    # ℃
        }
        return json.dumps(result, ensure_ascii=False)
    except (KeyError, IndexError):
        return json.dumps({"error": f"Could not parse weather data for '{city}'"})


@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast for the next N days.

    Args:
        city: City name (e.g., "Tokyo", "New York", "London")
        days: Number of forecast days (1-7, default 3)
    """
    if not 1 <= days <= 7:
        return json.dumps({"error": "days must be between 1 and 7"})

    url = f"{WTTR_URL}/{urllib.parse.quote(city)}?format=j1"
    data = _fetch_json(url)

    if data is None:
        return json.dumps({"error": f"Could not fetch forecast data for '{city}'"})

    try:
        weather_list = data["weather"][:days]
        forecast = []
        for day in weather_list:
            forecast.append({
                "date": day["date"],
                "max_temp": f"{day['maxtempC']}℃",
                "min_temp": f"{day['mintempC']}℃",
                "condition": day["hourly"][0]["weatherDesc"][0]["value"],
            })
        result = {"city": city, "forecast": forecast}
        return json.dumps(result, ensure_ascii=False)
    except (KeyError, IndexError):
        return json.dumps({"error": f"Could not parse forecast data for '{city}'"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    import uvicorn
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
