"""B07 系统组装：统一分析数据流（/analyze 的后端编排）。

严格顺序：
1. 获取工具结果：调用 5 个工具函数（当前环境/历史/天气/知识/视觉）+ B06 ML 预测；
2. 生成证据记录：统一 6 字段格式，source_type 区分 B01 状态 / B02 历史 /
   B03 天气 / B04 知识 / B06 ML 预测 / B06 视觉；
3. 形成结构化判断：规则函数根据证据生成 JSON（当前环境判断 / 近期趋势 /
   下一窗口预测 / 证据完整度）；
4. LLM 解释：结构化判断 + 证据摘要作为 Context 发给 LLM 生成自然语言解释；
5. 统一 run_id：UUID 贯穿全程，全部过程记录写入 SQLite analysis_runs 表。

任何一步失败都不会中断整条流水线：工具失败 → 证据 status=error；
LLM 失败 → explanation=None；存库失败 → 仅记录日志。
"""
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import db
import predict_model
import tools
from llm_client import LLMRequestError, chat
from vision_tool import analyze_camera_image

logger = logging.getLogger("analysis")

DEFAULT_LOCATION = "南京"
HISTORY_HOURS = 24
AUTO_KNOWLEDGE_QUERY = "锂电池热失控前兆与 BMS 管理规范"
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

# 证据记录的 6 个强制字段之一：source_type 阶段标注
S2T_STATE = "B01_STATE"  # 当前模拟状态（B01）
S2T_HISTORY = "B02_HISTORY"  # 历史数据（B02）
S2T_WEATHER = "B03_WEATHER"  # 天气（B03）
S2T_KNOWLEDGE = "B04_KNOWLEDGE"  # 知识库（B04）
S2T_ML = "B06_ML_PREDICT"  # ML 预测（B06）
S2T_VISION = "B06_VISION"  # YOLO 视觉（B06）

# 场景测试（前端下拉框 S1~S5，按验证指南定义）：
# 全部注入数据来自 scenarios.json 固定配方（当前环境数值 / 历史 48 点曲线 / 天气 24h 数值 /
# 天气与 YOLO 状态 / S1~S5 的 ML 物理阈值规则引擎覆盖），运行时禁止随机生成，保证每次运行结果完全一致。
SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios.json"
with open(SCENARIOS_PATH, encoding="utf-8") as _f:
    SCENARIO_CONFIG = json.load(_f)
SCENARIOS = {k: v["name"] for k, v in SCENARIO_CONFIG["scenarios"].items()}


def _evidence(source_type, tool_result):
    """把工具的 5 字段结果转换为统一的 6 字段证据记录。"""
    return {
        "source_type": source_type,
        "source": tool_result["source"],
        "time": tool_result["time"],
        "status": tool_result["status"],
        "result": tool_result["data"],
        "error": tool_result["error"],
    }


def _ml_predict_evidence(analysis_time):
    """B06 ML 预测证据：用 analysis_time 前最近 6 条数据构建预测窗口。"""
    t = analysis_time or datetime.now()
    now = datetime.now().strftime(TS_FORMAT)
    try:
        model_dict = predict_model.load_model()
        if analysis_time:
            rows = db.query_history(
                (t - timedelta(hours=4)).strftime(TS_FORMAT), t.strftime(TS_FORMAT), 5000
            )[-6:]
        else:
            rows = db.latest_samples(6)
        if len(rows) < 6:
            return {
                "source_type": S2T_ML, "source": "ML_MODEL", "time": now,
                "status": "error", "result": None,
                "error": f"该时刻前数据不足 6 条（实际 {len(rows)} 条），无法构建预测窗口",
            }
        r = predict_model.predict_attention(model_dict, rows)
        return {
            "source_type": S2T_ML, "source": "ML_MODEL", "time": now,
            "status": "ok",
            "result": {"window_end": rows[-1]["timestamp"], **r},
            "error": None,
        }
    except FileNotFoundError as exc:
        logger.warning("Model pkl missing: %s", exc)
        return {
            "source_type": S2T_ML, "source": "ML_MODEL", "time": now,
            "status": "unavailable", "result": None, "error": f"模型文件不存在：{exc}",
        }
    except Exception as exc:
        logger.exception("ML prediction evidence failed")
        return {
            "source_type": S2T_ML, "source": "ML_MODEL", "time": now,
            "status": "error", "result": None, "error": f"ML 预测失败：{exc}",
        }


def _history_from_config(cfg, scenario, end_time):
    """按 scenarios.json 的固定数值重建历史证据。

    数值逐点来自配置文件（温度/湿度/电压 48 点，30 分钟间隔），仅时间戳按分析时刻回推；
    不调用任何随机/三角函数生成数据。
    """
    n = SCENARIO_CONFIG["history"]["point_count"]
    interval = SCENARIO_CONFIG["history"]["interval_minutes"]
    end = end_time or datetime.now().replace(second=0, microsecond=0)
    times = [end - timedelta(minutes=interval * (n - 1 - i)) for i in range(n)]
    h = cfg["history"]
    samples = [
        {
            "timestamp": times[i].strftime(TS_FORMAT),
            "temperature": h["temperature"][i],
            "humidity": h["humidity"][i],
            "voltage": h["voltage"][i],
            "source_type": "SIMULATED",
        }
        for i in range(n)
    ]
    return {
        "count": n,
        "start": times[0].strftime(TS_FORMAT),
        "end": times[-1].strftime(TS_FORMAT),
        "samples": samples,
        "scenario_note": (
            f"场景 {scenario} 注入：{h['note']}历史（24h，30 分钟间隔，数值来自 scenarios.json）"
        ),
    }


def _weather_from_config(wcfg, scenario, location):
    """按 scenarios.json 的固定数值重建天气证据（24h 逐小时，数值固定，时间戳相对当前时刻）。"""
    n = len(wcfg["temperature_2m"])
    now = datetime.now()
    times = [(now + timedelta(hours=i)).strftime(TS_FORMAT) for i in range(n)]
    return {
        "provider": f"SIMULATED_{scenario}",
        "retrieved_at": now.strftime(TS_FORMAT),
        "location_name": location,
        "times": times,
        "temperature_2m": list(wcfg["temperature_2m"]),
        "relative_humidity_2m": list(wcfg["relative_humidity_2m"]),
        "precipitation_probability": list(wcfg["precipitation_probability"]),
        "scenario_note": f"场景 {scenario} 注入：{wcfg['note']}（数值来自 scenarios.json）",
    }


def _recompute_ml(evidence, samples, scenario):
    """历史被场景注入后，ML 预测窗口同步改用同一段合成历史（保持一致）。"""
    last6 = samples[-6:]
    for e in evidence:
        if e["source_type"] != S2T_ML:
            continue
        try:
            model_dict = predict_model.load_model()
            r = predict_model.predict_attention(model_dict, last6)
            e.update(
                status="ok", error=None,
                result={
                    "window_end": last6[-1]["timestamp"], **r,
                    "scenario_note": f"场景 {scenario}：预测窗口取自 scenarios.json 固定注入历史",
                },
            )
        except Exception as exc:
            logger.warning("ML recompute on scenario history failed: %s", exc)
            e.update(status="unavailable", result=None, error=f"ML 预测不可用：{exc}")
        break
    return evidence


# 物理阈值兜底规则引擎：字段注册表（字段名 -> 中文标签/单位/数值格式化）
_RULE_FIELD_META = {
    "hist_temp_delta": ("24h 温差", "℃", "{:+.1f}"),
    "current_temp": ("当前温度", "℃", "{:.1f}"),
    "hist_voltage_delta": ("24h 电压变化", "V", "{:+.2f}"),
    "current_voltage": ("当前电压", "V", "{:.2f}"),
}
_RULE_OPS = {"gt": ">", "lt": "<", "gte": "≥", "lte": "≤"}
# WATCH 兜底拦截的统一结论文案：车规级 Fail-Safe 安全拦截策略（由引擎统一追加）
_RULE_WATCH_TAIL = (
    "，触发车规级 Fail-Safe 安全拦截策略："
    "当前实测温度已越过安全红线，直接下发最高级别预警。"
)


def _rule_engine_values(hist, env):
    """从注入的历史曲线与当前环境计算阈值规则引擎的输入量（与 B01/B02 展示同源）。"""
    temps = [s["temperature"] for s in hist["samples"]]
    volts = [s["voltage"] for s in hist["samples"]]
    return {
        "hist_temp_delta": round(temps[-1] - temps[0], 2),
        "current_temp": env["temperature"],
        "hist_voltage_delta": round(volts[-1] - volts[0], 3),
        "current_voltage": env["voltage"],
    }


def _rule_cond_hit(cond, values):
    """单条件判定：abs=true 时按幅度（绝对值）比较，如「温差 < 5℃」。"""
    v = values[cond["field"]]
    if cond.get("abs"):
        v = abs(v)
    op = _RULE_OPS.get(cond["op"])
    if op == ">":
        return v > cond["value"]
    if op == "<":
        return v < cond["value"]
    if op == "≥":
        return v >= cond["value"]
    if op == "≤":
        return v <= cond["value"]
    return False


def _apply_rule_engine(mlcfg, hist, env, scenario):
    """物理阈值兜底规则引擎：按 scenarios.json 声明的阈值规则评估注入数据。

    规则按顺序评估，首条命中生效（危险规则须排在前面）；basis 用 {details}
    占位符自动填入实际触发的条件与实测值。全部未命中返回 None，由调用方
    保留模型原始输出。
    """
    values = _rule_engine_values(hist, env)
    for rule in mlcfg.get("rule_engine", {}).get("rules", []):
        conds = rule.get("all") or rule.get("any") or []
        hits = [_rule_cond_hit(c, values) for c in conds]
        matched = all(hits) if rule.get("all") else any(hits)
        if not matched:
            continue
        details = []
        for cond, hit in zip(conds, hits):
            if not hit:
                continue
            label, unit, fmt = _RULE_FIELD_META[cond["field"]]
            details.append(
                f"{label} {fmt.format(values[cond['field']])}{unit} "
                f"{_RULE_OPS[cond['op']]} {cond['value']:g}{unit} 阈值"
            )
        basis = rule["basis"].replace("{details}", "；".join(details))
        if rule["force"] == "WATCH":
            basis += _RULE_WATCH_TAIL
        return {
            "predicted_attention": rule["force"],
            "basis": basis,
            "engine": "threshold_fallback",
            "scenario_note": mlcfg.get(
                "scenario_note",
                f"场景 {scenario} 注入：物理阈值规则引擎兜底覆盖模型预测",
            ),
        }
    return None


def _apply_scenario(evidence, scenario, location, analysis_time):
    """按 scenarios.json 的固定配方注入证据行为（S1~S5 全部注入，禁止随机生成）。

    只替换本次分析的证据记录，绝不改动全局模拟状态（/toggle 仍由用户控制）。
    """
    if not scenario:
        return evidence
    cfg = SCENARIO_CONFIG["scenarios"][scenario]
    now = datetime.now().strftime(TS_FORMAT)

    # 1) 当前环境：JSON 固定数值（S1~S5 均为 State A 基准值）
    env = cfg["current_environment"]
    for e in evidence:
        if e["source_type"] == S2T_STATE:
            e.update(status="ok", source="SIMULATED", time=now, result=dict(env), error=None)

    # 2) 历史：JSON 固定 48 点曲线，ML 窗口同步重算（同一段数据，确定性输出）
    hist = _history_from_config(cfg, scenario, analysis_time)
    for e in evidence:
        if e["source_type"] == S2T_HISTORY:
            e.update(status="ok", result=hist, error=None)
    _recompute_ml(evidence, hist["samples"], scenario)

    # 3) 天气：JSON 固定数值（mode=ok）或超时（mode=timeout）
    wcfg = cfg["weather"]
    if wcfg["mode"] == "timeout":
        for e in evidence:
            if e["source_type"] == S2T_WEATHER:
                e.update(status="unavailable", result=None, error=wcfg["error_message"])
    else:
        w = _weather_from_config(wcfg, scenario, location)
        for e in evidence:
            if e["source_type"] == S2T_WEATHER:
                e.update(status="ok", time=now, result=w, error=None)

    # 4) 知识：JSON 指定检索主题（S4 = 电池高温风险，其余场景保持自动主题）
    kw_query = cfg["knowledge"].get("query")
    if kw_query:
        kw = tools.search_battery_knowledge(kw_query)
        for e in evidence:
            if e["source_type"] == S2T_KNOWLEDGE:
                e.update(source=kw["source"], time=kw["time"], status=kw["status"],
                         result=kw["data"], error=kw["error"])
                if kw["status"] == "ok":
                    e["result"]["scenario_note"] = f"场景 {scenario} 注入：知识检索切换为『{kw_query}』"

    # 5) 视觉：JSON 声明 error 时覆盖为固定失败（S5 图片缺失）
    vcfg = cfg["vision"]
    if vcfg["status"] == "error":
        for e in evidence:
            if e["source_type"] == S2T_VISION:
                e.update(status="error", result=None, error=vcfg["error_message"])

    # 6) ML：JSON 声明 rule_override 时启用物理阈值兜底规则引擎（S1~S5 全部注入）
    # 阈值规则按注入的历史曲线与当前环境实测值判定，命中才覆盖模型输出；
    # 全部未命中则保留模型原始输出（修复特征量纲变化导致模型超训练分布的误判）。
    mlcfg = cfg["ml"]
    if mlcfg["mode"] == "rule_override":
        forced = _apply_rule_engine(mlcfg, hist, env, scenario)
        for e in evidence:
            if e["source_type"] == S2T_ML and e["status"] == "ok":
                if forced:
                    e["result"].update(forced)
                else:
                    e["result"]["scenario_note"] = (
                        f"场景 {scenario} 注入：阈值规则引擎未命中，保留模型原始输出"
                    )
    return evidence


def _safe_evidence(source_type, source, fn, *args, **kwargs):
    """带保护的工具采集：工具抛任何异常都转为 error 证据，绝不向外抛（S5 降级不崩溃）。"""
    now = datetime.now().strftime(TS_FORMAT)
    try:
        return _evidence(source_type, fn(*args, **kwargs))
    except Exception as exc:
        logger.exception("Evidence tool raised unexpected error (source_type=%s)", source_type)
        return {
            "source_type": source_type, "source": source, "time": now,
            "status": "error", "result": None, "error": f"工具执行异常：{exc}",
        }


def collect_evidence(analysis_time, location):
    """步骤 1+2：调用全部工具，产出统一的证据记录列表（6 条）。

    每个工具都经 _safe_evidence 保护：工具异常 → error 证据，流水线不中断。
    """
    evidence = []

    # B01 当前环境（模拟状态，与 analysis_time 无关）
    evidence.append(_safe_evidence(S2T_STATE, "SIMULATED", tools.get_current_environment))

    # B02 历史：默认调工具；指定 analysis_time 时按该时刻回溯 24h（同样结构）
    if analysis_time:
        start = (analysis_time - timedelta(hours=HISTORY_HOURS)).strftime(TS_FORMAT)
        end = analysis_time.strftime(TS_FORMAT)
        try:
            samples = db.query_history(start, end, 5000)
            evidence.append({
                "source_type": S2T_HISTORY, "source": "SIMULATED",
                "time": datetime.now().strftime(TS_FORMAT), "status": "ok",
                "result": {"count": len(samples), "start": start, "end": end, "samples": samples},
                "error": None,
            })
        except Exception as exc:
            logger.exception("History query failed in evidence collection")
            evidence.append({
                "source_type": S2T_HISTORY, "source": "SIMULATED",
                "time": datetime.now().strftime(TS_FORMAT),
                "status": "error", "result": None, "error": f"历史数据查询失败：{exc}",
            })
    else:
        evidence.append(_safe_evidence(S2T_HISTORY, "SIMULATED",
                                       tools.get_environment_history, HISTORY_HOURS))

    # B03 天气（外部 API，与 analysis_time 无关）
    evidence.append(_safe_evidence(S2T_WEATHER, "PUBLIC", tools.get_weather_forecast, location))

    # B04 知识库（自动检索温湿度管理要点）
    evidence.append(_safe_evidence(S2T_KNOWLEDGE, "KNOWLEDGE_BASE",
                                   tools.search_battery_knowledge, AUTO_KNOWLEDGE_QUERY))

    # B06 ML 预测（自带保护）+ 视觉
    evidence.append(_ml_predict_evidence(analysis_time))
    evidence.append(_safe_evidence(S2T_VISION, "YOLO_CAMERA", analyze_camera_image))

    return evidence


def make_judgment(evidence):
    """步骤 3：规则函数，根据证据生成结构化判断 JSON。"""
    by = {e["source_type"]: e for e in evidence}

    # --- 当前环境状态判断 ---
    env = by.get(S2T_STATE, {})
    if env.get("status") == "ok":
        r = env["result"]
        temp = r["temperature"]
        hum = r["humidity"]
        temp_j = "偏低" if temp < 18 else ("偏高" if temp > 30 else "适宜")
        hum_j = "偏干" if hum < 40 else ("偏湿" if hum > 70 else "适宜")
        current = {
            "state": r["name"],
            "temperature": temp,
            "humidity": hum,
            "voltage": r["voltage"],
            "assessment": f"温度 {temp}°C（{temp_j}），湿度 {hum}%（{hum_j}），电压 {r['voltage']} V",
        }
    else:
        current = {"state": None, "temperature": None, "humidity": None, "voltage": None,
                   "assessment": f"当前环境数据不可用：{env.get('error')}"}

    # --- 近期趋势 ---
    hist = by.get(S2T_HISTORY, {})
    if hist.get("status") == "ok" and hist["result"] and hist["result"].get("count", 0) >= 2:
        temps = [s["temperature"] for s in hist["result"]["samples"]]
        delta = round(temps[-1] - temps[0], 2)
        direction = "上升" if delta > 1 else ("下降" if delta < -1 else "平稳")
        trend = {
            "hours": HISTORY_HOURS,
            "count": len(temps),
            "temperature_delta": delta,
            "min_temperature": round(min(temps), 2),
            "max_temperature": round(max(temps), 2),
            "assessment": (
                f"过去 {HISTORY_HOURS} 小时温度整体{direction}（Δ{delta:+.2f}°C），"
                f"区间 {min(temps):.2f}~{max(temps):.2f}°C"
            ),
        }
    else:
        trend = {"assessment": f"历史数据不可用或不足：{hist.get('error')}"}

    # --- 证据冲突 / 不一致检测（S4：预报与历史趋势相反；S3：高温预警与平稳历史不一致） ---
    conflicts = []
    we = by.get(S2T_WEATHER, {})
    if (hist.get("status") == "ok" and hist["result"] and hist["result"].get("count", 0) >= 2
            and we.get("status") == "ok" and we["result"] and we["result"].get("temperature_2m")):
        h_temps = [s["temperature"] for s in hist["result"]["samples"]]
        hist_delta = h_temps[-1] - h_temps[0]
        w_temps = we["result"]["temperature_2m"]
        w_delta = w_temps[-1] - w_temps[0]
        if (hist_delta < -1 and w_delta > 1) or (hist_delta > 1 and w_delta < -1):
            conflicts.append(
                f"B03 天气预报走向（{w_delta:+.1f}°C）与 B02 历史趋势（{hist_delta:+.1f}°C）相反"
            )
        w_max, h_max = max(w_temps), max(h_temps)
        if w_max >= 35 and h_max < 30:
            conflicts.append(
                f"B03 天气预报发出高温预警（未来 24h 最高 {w_max:.0f}°C），"
                f"与 B02 历史平稳状态（最高 {h_max:.0f}°C）不一致"
            )

    # --- 下一窗口预测（ML 模型，规则兜底字段 basis 随证据带出） ---
    ml = by.get(S2T_ML, {})
    if ml.get("status") == "ok":
        prediction = {
            "predicted_attention": ml["result"].get("predicted_attention"),
            "model_version": ml["result"].get("model_version"),
            "window_end": ml["result"].get("window_end"),
        }
        if ml["result"].get("basis"):
            prediction["basis"] = ml["result"]["basis"]
        if ml["result"].get("engine"):
            prediction["engine"] = ml["result"]["engine"]
    else:
        prediction = {"predicted_attention": "UNKNOWN", "error": ml.get("error")}

    # --- 证据完整度 ---
    total = len(evidence)
    ok = sum(1 for e in evidence if e["status"] == "ok")
    unavailable = sum(1 for e in evidence if e["status"] == "unavailable")
    errors = total - ok - unavailable
    completeness = {
        "total": total,
        "ok": ok,
        "unavailable": unavailable,
        "error": errors,
        "ratio": round(ok / total, 3),
    }

    judgment = {
        "current_environment": current,
        "recent_trend": trend,
        "next_window_prediction": prediction,
        "evidence_completeness": completeness,
        "evidence_conflicts": conflicts,
    }
    # 证据不完整时写入确定性的限制声明（不依赖 LLM，UI/报告直接可见）
    if completeness["ok"] < completeness["total"]:
        judgment["limitation"] = (
            f"证据不完整（{completeness['ok']}/{completeness['total']}："
            f"unavailable {completeness['unavailable']} · error {completeness['error']}），"
            "本次判断受限，结论仅供参考"
        )
    return judgment


def _digest(evidence):
    """证据摘要：发给 LLM 前按类型精简，避免塞入全量数据。"""
    r = evidence["result"]
    t = evidence["source_type"]
    if evidence["status"] != "ok":
        return {"source_type": t, "status": evidence["status"], "error": evidence["error"]}
    if t == S2T_STATE:
        return {"source_type": t, "status": "ok", "result": {
            "name": r["name"], "temperature": r["temperature"], "humidity": r["humidity"],
            "voltage": r["voltage"], "source_type": r.get("source_type"),
        }}
    if t == S2T_HISTORY:
        temps = [s["temperature"] for s in r["samples"]]
        if not temps:
            return {"source_type": t, "status": "ok", "count": 0,
                    "note": "该时间段内没有历史数据"}
        return {"source_type": t, "status": "ok", "count": r["count"],
                "temperature_range": [round(min(temps), 2), round(max(temps), 2)],
                "temperature_delta": round(temps[-1] - temps[0], 2)}
    if t == S2T_WEATHER:
        temps = r["temperature_2m"]
        preps = r["precipitation_probability"]
        return {"source_type": t, "status": "ok", "points": len(r["times"]),
                "temperature_range": ([round(min(temps), 1), round(max(temps), 1)]
                                      if temps else None),
                "precipitation_max": (round(max(preps), 1) if preps else None),
                "provider": r.get("provider"), "location": r.get("location_name")}
    if t == S2T_KNOWLEDGE:
        return {"source_type": t, "status": "ok", "count": r["count"],
                "top_sources": [f"{x['source']} 第{x['paragraph']}段" for x in r["results"]],
                "top_texts": [x["text"][:160] for x in r["results"]]}
    if t == S2T_ML:
        return {"source_type": t, "status": "ok", "result": r}
    if t == S2T_VISION:
        return {"source_type": t, "status": "ok", "result": r}
    return {"source_type": t, "status": "ok", "result": r}


def explain_with_llm(judgment, evidence):
    """步骤 4：结构化判断 + 证据摘要作为 Context，让 LLM 生成自然语言解释。"""
    system = (
        "你是锂电池储能安全分析助手。系统会给你结构化判断与证据摘要，"
        "请生成一段 150~300 字的中文分析报告，顺序：当前环境结论 → 近期趋势 → "
        "下一窗口预测 → 证据完整度与注意事项 → 处置建议。只能使用给出的数据，不得编造。"
        "硬性要求：1) 证据之间存在矛盾（如 evidence_conflicts 非空、天气预报方向与历史趋势相反）时，"
        "必须明确指出该分歧；2) 证据完整度未达 100%（存在 unavailable 或 error 证据）时，"
        "报告必须逐项列出缺失的证据（如「缺失 B03 天气证据」「缺失 B06 视觉证据」），"
        "并且全文最后一句必须一字不差地以「本次判断受限」结尾，"
        "不得省略、不得换成其他表述、不得在结尾后再补充内容，"
        "正文中除最后一句外也不得重复声明「判断受限」；"
        "3) 证据完整度达到 100%（六项证据全部 status=ok）时，全文严禁出现「受限」二字，"
        "不得声称判断受限或结论仅供参考，直接给出明确结论；"
        "4) evidence_conflicts 为空、且天气预报方向与历史趋势一致时，"
        "严禁声称存在分歧或矛盾，不得使用「分歧」「矛盾」等字眼；"
        "5) 必须依据 B04 知识库检索结果，在报告末尾给出针对当前异常的具体处置建议"
        "（如「建议立即停车并疏散人员」「建议立即停止充电」「建议切断高压继电器并撤离」），"
        "处置措施只能来自知识库内容，不得编造知识库未给出的措施；"
        "当前无异常时可给出保持监测、按规程运维等常规建议；"
        "证据完整度未达 100% 时，处置建议须放在最后一句「本次判断受限」之前；"
        "6) 若分析涉及或有人问及数据频率，必须说明：本系统为云端离线复盘，非车端实时控制；"
        "7) 当 next_window_prediction 含 engine=threshold_fallback（温度或电压触发规则引擎兜底）时，"
        "必须明确说明该 WATCH 预警来自「Fail-Safe 安全防线」的规则拦截，不得表述为模型直接预测；"
        "处置建议仍按规则 5 从 B04 知识库提取（如切断高压、疏散人员）。"
    )
    context = json.dumps(
        {"judgment": judgment, "evidence_digest": [_digest(e) for e in evidence]},
        ensure_ascii=False,
        default=str,
    )
    message = chat([
        {"role": "system", "content": system},
        {"role": "user", "content": context},
    ])
    return message["content"]


def _enforce_no_limitation_when_complete(explanation, judgment):
    """证据完整度 100% 时的兜底：删除 LLM 输出中任何含「受限」的句子，防止自相矛盾。"""
    if not explanation or "受限" not in explanation:
        return explanation
    completeness = judgment.get("evidence_completeness", {})
    if completeness.get("ok") == completeness.get("total") and completeness.get("total"):
        kept = [s for s in explanation.split("。") if "受限" not in s]
        logger.warning("LLM explanation contained '受限' despite complete evidence; stripped")
        return "。".join(kept)
    return explanation


def run_analysis(analysis_time=None, location=DEFAULT_LOCATION, scenario=None):
    """步骤 5：完整流水线，run_id 贯穿，全部过程写入 analysis_runs 表。"""
    run_id = str(uuid.uuid4())
    # 显式指定时刻才按该时刻回溯（历史 + ML 窗口均截至该时刻）；
    # 缺省（None）走"最新数据"路径：历史取最近 24h，ML 取库中最新 6 条
    explicit_time = analysis_time is not None
    analysis_time = analysis_time or datetime.now().replace(second=0, microsecond=0)
    analysis_time_str = analysis_time.strftime(TS_FORMAT)

    try:
        evidence = _apply_scenario(
            collect_evidence(analysis_time if explicit_time else None, location),
            scenario,
            location,
            analysis_time,
        )
    except Exception:
        # 兜底：采集/注入环节任何未预期异常都不中断流水线，降级为全 error 证据
        logger.exception("Evidence collection crashed; degrading to error-only evidence")
        evidence = [
            {
                "source_type": t, "source": "ANALYSIS_FALLBACK",
                "time": analysis_time_str, "status": "error", "result": None,
                "error": "证据采集环节异常，本次证据不可用",
            }
            for t in (S2T_STATE, S2T_HISTORY, S2T_WEATHER, S2T_KNOWLEDGE, S2T_ML, S2T_VISION)
        ]
    judgment = make_judgment(evidence)
    if scenario:
        judgment["scenario"] = scenario
        judgment["scenario_name"] = SCENARIOS.get(scenario, scenario)

    try:
        explanation = explain_with_llm(judgment, evidence)
    except LLMRequestError as exc:
        logger.warning("LLM explanation failed (kind=%s): %s", exc.kind, exc)
        explanation = None
    except Exception:
        logger.exception("LLM explanation failed")
        explanation = None
    if explanation:
        explanation = _enforce_no_limitation_when_complete(explanation, judgment)

    created_at = datetime.now().strftime(TS_FORMAT)
    try:
        db.insert_analysis_run(
            run_id,
            analysis_time_str,
            location,
            json.dumps(evidence, ensure_ascii=False, default=str),
            json.dumps(judgment, ensure_ascii=False, default=str),
            explanation,
            created_at,
        )
        logger.info("Analysis run %s saved (evidence=%d, ok=%d)", run_id, len(evidence),
                    judgment["evidence_completeness"]["ok"])
    except Exception:
        logger.exception("Failed to save analysis run %s", run_id)

    return {
        "run_id": run_id,
        "analysis_time": analysis_time_str,
        "location": location,
        "scenario": scenario,
        "evidence": evidence,
        "judgment": judgment,
        "explanation": explanation,
    }


def run_to_markdown(run):
    """把一条分析记录渲染为可读的 Markdown 运行报告（证据链固化，供导出下载）。

    run：db.get_analysis_run 返回的字典（evidence/judgment 为 JSON 字符串）。
    """
    evidence = json.loads(run["evidence"]) if isinstance(run["evidence"], str) else run["evidence"]
    judgment = json.loads(run["judgment"]) if isinstance(run["judgment"], str) else run["judgment"]
    j = judgment or {}
    env = j.get("current_environment") or {}
    trend = j.get("recent_trend") or {}
    pred = j.get("next_window_prediction") or {}
    comp = j.get("evidence_completeness") or {}

    lines = [
        "# GreenMate 分析运行报告",
        "",
        f"- **run_id**：{run['run_id']}",
        f"- **分析时刻**：{run['analysis_time']}",
        f"- **地点**：{run['location']}",
        f"- **场景**：{j.get('scenario') or '—'}（{j.get('scenario_name') or '无'}）",
        f"- **创建时间**：{run['created_at']}",
        "",
        "## 结构化判断",
        "",
        f"- **当前环境判断**：{env.get('assessment') or '（无）'}",
    ]
    if env.get("temperature") is not None:
        lines.append(f"  - 温度 {env['temperature']} °C / 湿度 {env['humidity']} % / 电压 {env['voltage']} V")
    lines += [
        f"- **近期趋势**：{trend.get('assessment') or '（无）'}",
        f"- **下一窗口预测**：{pred.get('predicted_attention') or 'UNKNOWN'}"
        f"（模型 {pred.get('model_version') or '—'}，窗口截至 {pred.get('window_end') or '—'}）",
        f"- **证据完整度**：{comp.get('ok', 0)}/{comp.get('total', 0)}"
        f"（{round((comp.get('ratio') or 0) * 100)}%）"
        f"· 不可用 {comp.get('unavailable', 0)} · 失败 {comp.get('error', 0)}",
    ]
    if pred.get("basis"):
        lines.append(f"- **提升依据**：{pred['basis']}")
    lines.append("- **证据冲突/分歧**：")
    for c in j.get("evidence_conflicts") or []:
        lines.append(f"  - ⚠ {c}")
    if not (j.get("evidence_conflicts") or []):
        lines.append("  - （无）")

    lines += ["", "## 证据列表", ""]
    mark = {"ok": "✅", "unavailable": "⚠️", "error": "❌"}
    for e in evidence or []:
        st = e.get("status")
        lines += [
            f"### {e.get('source_type')} —— {mark.get(st, '❓')} {st}",
            "",
            f"- 来源：{e.get('source')}　时间：{e.get('time')}　状态：{st}",
        ]
        if e.get("error"):
            lines.append(f"- 错误：{e['error']}")
        if e.get("result") is not None:
            lines += ["- 原始结果：", "", "```json",
                      json.dumps(e["result"], ensure_ascii=False, indent=2, default=str), "```"]
        lines.append("")

    lines += ["## LLM 解释", "", run.get("explanation") or "（缺失：LLM 调用失败或不可用）", ""]
    return "\n".join(lines)
