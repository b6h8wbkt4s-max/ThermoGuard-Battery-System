"""标准工具函数（B05）。

所有函数统一返回字典，五个字段恒存在：
- status: 'ok'（成功）/ 'unavailable'（暂时不可用：网络超时、外部服务失败）/ 'error'（本地错误）
- source: 数据来源标记（SIMULATED / PUBLIC / KNOWLEDGE_BASE）
- time:   工具执行时间（YYYY-MM-DD HH:MM:SS）
- data:   业务数据（dict 或 None）
- error:  错误说明（无错误时为 None）

任何函数在异常时都返回统一字典，绝不向外抛异常（可独立运行）。
"""
import logging
import sys
from datetime import datetime, timedelta

import db
import state
from rag import load_knowledge, retrieve
from weather_client import LocationNotFound, WeatherError, geocode, get_forecast

logger = logging.getLogger("tools")


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _result(status, source, data, error=None):
    return {"status": status, "source": source, "time": _now(), "data": data, "error": error}


def get_current_environment():
    """返回当前环境状态（本地模拟数据）。"""
    current = state.get_current_state()
    return _result(
        "ok",
        "SIMULATED",
        {
            "name": current["name"],
            "temperature": current["temperature"],
            "humidity": current["humidity"],
            "voltage": current["voltage"],
            "source_type": "SIMULATED",
        },
    )


def get_environment_history(hours=24):
    """查询最近 hours 小时的历史数据（SQLite，每 30 分钟一个点）。"""
    try:
        hours = float(hours)
    except (TypeError, ValueError):
        return _result("error", "SIMULATED", None, f"hours 参数无效：{hours!r}")
    if hours <= 0:
        return _result("error", "SIMULATED", None, "hours 必须为正数")
    try:
        end = datetime.now().replace(second=0, microsecond=0)
        start = end - timedelta(hours=hours)
        samples = db.query_history(
            start.strftime("%Y-%m-%d %H:%M:%S"),
            end.strftime("%Y-%m-%d %H:%M:%S"),
            limit=5000,
        )
    except Exception as exc:
        logger.exception("Tool get_environment_history failed")
        return _result("error", "SIMULATED", None, f"历史数据查询失败：{exc}")
    return _result(
        "ok",
        "SIMULATED",
        {
            "count": len(samples),
            "start": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": end.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": samples,
        },
    )


def get_weather_forecast(location):
    """获取指定地点未来天气（Open-Meteo geocoding 动态解析任意城市名，PUBLIC 外部数据）。"""
    if not location or not location.strip():
        return _result("error", "PUBLIC", None, "地点不能为空")
    location = location.strip()
    try:
        latitude, longitude, resolved = geocode(location)
        forecast = get_forecast(location_name=resolved, latitude=latitude, longitude=longitude)
    except LocationNotFound as exc:
        # 地点无法解析：用户输入问题
        return _result("error", "PUBLIC", None, str(exc))
    except WeatherError as exc:
        # 网络超时/外部服务失败：暂时不可用
        return _result("unavailable", "PUBLIC", None, str(exc))
    except Exception as exc:
        logger.exception("Tool get_weather_forecast failed")
        return _result("error", "PUBLIC", None, f"天气获取失败：{exc}")
    return _result("ok", "PUBLIC", forecast)


def search_battery_knowledge(query):
    """检索锂电池安全知识库（TF-IDF，返回最相关的 2 个 chunk）。"""
    if not query or not query.strip():
        return _result("error", "KNOWLEDGE_BASE", None, "检索词不能为空")
    try:
        chunks = load_knowledge()
        if not chunks:
            return _result("error", "KNOWLEDGE_BASE", None, "知识库为空或加载失败")
        top = retrieve(query.strip(), k=2)
    except Exception as exc:
        logger.exception("Tool search_battery_knowledge failed")
        return _result("error", "KNOWLEDGE_BASE", None, f"知识检索失败：{exc}")
    results = [
        {
            "score": round(score, 4),
            "source": chunk["source"],
            "paragraph": chunk["paragraph"],
            "text": chunk["text"],
        }
        for score, chunk in top
    ]
    return _result("ok", "KNOWLEDGE_BASE", {"count": len(results), "results": results})


if __name__ == "__main__":
    # 独立运行演示：每个工具都能单独执行且不抛异常
    print("== get_current_environment ==")
    print(get_current_environment())
    print("== get_environment_history(24) ==")
    r = get_environment_history(24)
    print({k: (v if k != "data" else f"<{len(v['samples'])} samples>") for k, v in r.items()})
    print("== get_weather_forecast(任意城市，缺省南京) ==")
    city = sys.argv[1] if len(sys.argv) > 1 else "南京"
    r = get_weather_forecast(city)
    print({k: (v if k != "data" else f"<{len(v['times'])} points>") for k, v in r.items()})
    print("== search_battery_knowledge('锂电池热失控前兆有哪些') ==")
    print(search_battery_knowledge("锂电池热失控前兆有哪些"))
