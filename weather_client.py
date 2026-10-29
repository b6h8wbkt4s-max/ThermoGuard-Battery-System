import logging
from datetime import datetime

import requests

logger = logging.getLogger("weather_client")

PROVIDER = "Open-Meteo"
TIMEOUT_SECONDS = 10

# 未指定地点时的默认城市（/weather 页面与 /analyze 缺省值）
DEFAULT_LOCATION = "南京"


class WeatherError(Exception):
    """天气获取失败。message 为面向用户的提示，原始异常通过 __cause__ 保留在日志中。"""


class LocationNotFound(WeatherError):
    """地点无法通过 geocoding 解析（用户输入问题，区别于网络故障）。"""


def geocode(location):
    """Open-Meteo geocoding：城市名 -> (纬度, 经度, 官方名称)。支持任意中文城市名。

    解析失败抛 LocationNotFound；网络故障抛 WeatherError。
    """
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "zh", "format": "json"},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        results = (resp.json() or {}).get("results") or []
    except (requests.Timeout, requests.ConnectionError) as exc:
        logger.error("Geocoding request failed (timeout/network)")
        raise WeatherError("地点解析失败：网络超时或连接失败") from exc
    except requests.RequestException as exc:
        logger.error("Geocoding request failed: %s", exc)
        raise WeatherError("地点解析失败：服务异常") from exc
    if not results:
        raise LocationNotFound(f"无法解析地点：{location}（请提供城市名，如“北京”“上海”）")
    top = results[0]
    logger.info(
        "Geocoded %s -> %s (%.4f, %.4f)",
        location, top.get("name"), top["latitude"], top["longitude"],
    )
    return top["latitude"], top["longitude"], top.get("name") or location


def get_forecast(location_name=DEFAULT_LOCATION, latitude=None, longitude=None, days=3):
    """请求 Open-Meteo 未来天气（逐小时温度/湿度/降水概率）。

    latitude/longitude 缺省时先用 geocoding 按城市名动态解析（支持任意城市）。
    返回 dict 含 provider 与 retrieved_at（获取时间），供页面标注数据来源。
    """
    if latitude is None or longitude is None:
        latitude, longitude, location_name = geocode(location_name)

    retrieved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability",
                "timezone": "Asia/Shanghai",
                "forecast_days": days,
            },
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.Timeout, requests.ConnectionError) as exc:
        logger.error("Weather API request failed (timeout/network)")
        raise WeatherError("网络超时或连接失败") from exc
    except requests.RequestException as exc:
        logger.error("Weather API request failed: %s", exc)
        raise WeatherError("服务异常") from exc

    hourly = data.get("hourly", {})
    times = hourly.get("time") or []
    if not times:
        logger.error("Weather API returned empty hourly data")
        raise WeatherError("返回数据为空")
    logger.info(
        "Weather fetched (provider=%s, location=%s, %d hourly points)",
        PROVIDER, location_name, len(times),
    )
    return {
        "source_type": "PUBLIC",
        "provider": PROVIDER,
        "retrieved_at": retrieved_at,
        "location_name": location_name,
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": data.get("timezone"),
        "units": hourly.get("hourly_units", {}),
        "times": times,
        "temperature_2m": hourly.get("temperature_2m", []),
        "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
        "precipitation_probability": hourly.get("precipitation_probability", []),
    }
