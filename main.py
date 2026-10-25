import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

import analysis

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

import agent
import db
import state
from llm_client import LLMRequestError, ask_llm
from rag import build_rag_prompt, load_knowledge, retrieve
from weather_client import DEFAULT_LOCATION, LocationNotFound, WeatherError, get_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    logger.info("Database initialized at %s", db.DB_PATH)
    load_knowledge()
    yield


app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", state.state_data(state.get_current_key()))


@app.post("/toggle")
async def toggle():
    key = state.toggle_state()
    logger.info("State switched to %s", key)
    return state.state_data(key)


# 硬件边缘节点启动时刻（模拟 STM32 边缘网关，供心跳接口计算 uptime）
_EDGE_NODE_STARTED = datetime.now()


# 心跳判定：sensor.db 最新样本超过该秒数未更新，视为边缘节点离线（红灯）。
# Fetcher 每 3 秒拉取一次，5 秒窗口足以确认它是否真的在推送数据。
_HEARTBEAT_STALE_SECONDS = 5


@app.get("/api/hardware_heartbeat")
async def hardware_heartbeat():
    """硬件边缘节点心跳：轮询 sensor.db 最新样本，最近 5 秒内更新→online: true（绿灯），否则 online: false（红灯）。"""
    latest = None
    try:
        rows = db.latest_samples(1)
        if rows:
            latest = rows[0]
    except Exception:
        logger.exception("Heartbeat latest sample query failed")

    online = False
    fresh_seconds = None
    if latest:
        try:
            ts = datetime.strptime(latest["timestamp"], "%Y-%m-%d %H:%M:%S")
            fresh_seconds = int((datetime.now() - ts).total_seconds())
            online = fresh_seconds <= _HEARTBEAT_STALE_SECONDS
        except ValueError:
            logger.warning("Heartbeat: unparsable sample timestamp %r", latest["timestamp"])
    return {
        "node_id": "STM32-EDGE-01",
        "online": online,
        "status": "online" if online else "offline",
        "uptime_seconds": int((datetime.now() - _EDGE_NODE_STARTED).total_seconds()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "SIMULATED",
        "latest_sample": latest,
        "fresh_seconds": fresh_seconds,
        "stale_after_seconds": _HEARTBEAT_STALE_SECONDS,
    }


@app.post("/onenet/webhook")
async def onenet_webhook(request: Request):
    """OneNET 物联网平台数据推送：解析 temperature / voltage / humidity 并入库。

    请求体（JSON 扁平键）：
    {"temperature": 25.5, "voltage": 3.7, "humidity": 50.0, "timestamp": "2026-11-04 10:00:00"（可选）}
    三个数值字段均为必填；入库 source_type = ONENET；缺 timestamp 时按当前时刻记录。
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON"})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象"})

    def _num(key):
        v = body.get(key)
        if v is None:
            raise ValueError(f"缺少字段：{key}")
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"字段 {key} 不是有效数字：{v!r}")

    try:
        temperature = _num("temperature")
        voltage = _num("voltage")
        humidity = _num("humidity")
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    ts = str(body.get("timestamp") or "").strip()
    timestamp = ts if ts else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        row_id = db.insert_sample(timestamp, temperature, humidity, voltage, "ONENET")
    except Exception:
        logger.exception("OneNET webhook insert failed")
        return JSONResponse(status_code=502, content={"error": "数据入库失败"})
    logger.info(
        "OneNET webhook stored sample (id=%s, temp=%.1f, volt=%.2f, hum=%.1f)",
        row_id, temperature, voltage, humidity,
    )
    return {
        "status": "ok",
        "id": row_id,
        "timestamp": timestamp,
        "temperature": temperature,
        "voltage": voltage,
        "humidity": humidity,
        "source_type": "ONENET",
    }


def _llm_error_response(exc, tag):
    """LLM 调用失败：终端打印分类诊断 + 返回统一的 502。"""
    cause = exc.__cause__ or exc
    # 终端打印分类后的详细错误，便于直接区分密钥 / 网络问题
    if exc.kind == "auth":
        print(f"[{tag}] 密钥错误或未配置（请检查 .env 中 LLM_API_KEY）：{cause}", flush=True)
    elif exc.kind == "network":
        base_url = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
        print(f"[{tag}] 网络问题（连接失败/超时），请求地址 {base_url}/chat/completions：{cause}", flush=True)
    else:
        print(f"[{tag}] 大模型调用失败（{exc.kind}）：{cause}", flush=True)
    # 完整 traceback 同时写入控制台和 app.log
    logger.exception("LLM API call failed (endpoint=%s, kind=%s)", tag, exc.kind)
    return JSONResponse(status_code=502, content={"error": f"API调用失败：{exc}"})


@app.post("/ask")
async def ask(request: Request):
    # 自由提问：JSON body {"question": "..."}；页面 A/B 按钮空 POST 时沿用固定问题
    question = state.FIXED_QUESTION
    raw = await request.body()
    if raw:
        try:
            body = json.loads(raw)
        except Exception:
            logger.warning("Unparsable JSON body in /ask: %r", raw[:200])
            body = {}
        if isinstance(body, dict):
            custom = (body.get("question") or "").strip()
            if custom:
                question = custom

    current = state.get_current_state()
    try:
        answer, tool_calls = agent.run_agent(question)
    except LLMRequestError as exc:
        return _llm_error_response(exc, "ask")
    except Exception as exc:
        print(f"[ask] 未预期的服务器内部错误：{exc!r}", flush=True)
        logger.exception("Unexpected error handling /ask (state=%s)", state.get_current_key())
        return JSONResponse(status_code=502, content={"error": "API调用失败：服务器内部错误"})
    logger.info(
        "Ask succeeded (state=%s, question=%s, answer=%d chars, tool_calls=%d)",
        state.get_current_key(),
        question[:40],
        len(answer),
        tool_calls,
    )
    return {
        "state_name": current["name"],
        "temperature": current["temperature"],
        "humidity": current["humidity"],
        "voltage": current["voltage"],
        "question": question,
        "answer": answer,
        "tool_calls": tool_calls,
    }


@app.post("/agent/chat")
async def agent_chat(request: Request):
    """AI 助手主控台：自由提问，LLM 按 TOOL_DEFINITIONS 自动选择工具并完成闭环。"""
    raw = await request.body()
    if raw:
        try:
            body = json.loads(raw)
        except Exception:
            logger.warning("Unparsable JSON body in /agent/chat: %r", raw[:200])
            return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON"})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象"})
        question = (body.get("question") or "").strip()
    else:
        question = ""
    if not question:
        return JSONResponse(status_code=400, content={"error": "问题不能为空"})

    try:
        answer, tool_calls = agent.run_agent(question)
    except LLMRequestError as exc:
        return _llm_error_response(exc, "agent")
    except Exception:
        logger.exception("Unexpected error handling /agent/chat")
        return JSONResponse(status_code=502, content={"error": "API调用失败：服务器内部错误"})
    logger.info(
        "Agent chat succeeded (question=%s, answer=%d chars, tool_calls=%d)",
        question[:40],
        len(answer),
        tool_calls,
    )
    return {"question": question, "answer": answer, "tool_calls": tool_calls}


@app.get("/history")
async def history(start: str | None = None, end: str | None = None, limit: int = 5000):
    # datetime-local 控件提交 "YYYY-MM-DDTHH:MM"，转成与库内一致的 "YYYY-MM-DD HH:MM:SS"
    start = start.replace("T", " ") if start else None
    end = end.replace("T", " ") if end else None
    limit = max(1, min(limit, 5000))
    try:
        samples = db.query_history(start, end, limit)
    except Exception:
        logger.exception("History query failed (start=%s, end=%s)", start, end)
        return JSONResponse(status_code=500, content={"error": "查询历史数据失败"})
    return {"count": len(samples), "samples": samples}


@app.get("/weather")
async def weather(location: str = DEFAULT_LOCATION):
    """B03 天气查询：?location=城市名 动态解析任意城市，缺省为南京。"""
    try:
        forecast = get_forecast(location_name=location.strip() or DEFAULT_LOCATION)
    except LocationNotFound as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except WeatherError:
        logger.exception("Weather fetch failed")
        return JSONResponse(status_code=502, content={"error": "天气获取失败"})
    except Exception:
        logger.exception("Unexpected error handling /weather")
        return JSONResponse(status_code=502, content={"error": "天气获取失败"})
    return forecast


@app.post("/rag")
async def rag(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON"})
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "问题不能为空"})

    top = retrieve(question, k=2)
    try:
        answer = ask_llm(build_rag_prompt(question, top))
    except LLMRequestError as exc:
        logger.exception("RAG LLM call failed")
        return JSONResponse(status_code=502, content={"error": f"API调用失败：{exc}"})
    except Exception:
        logger.exception("Unexpected error handling /rag")
        return JSONResponse(status_code=502, content={"error": "API调用失败：服务器内部错误"})
    sources = [
        {"source": chunk["source"], "paragraph": chunk["paragraph"], "text": chunk["text"]}
        for _, chunk in top
    ]
    logger.info("RAG answered (question=%s, sources=%d)", question[:40], len(sources))
    return {"question": question, "sources": sources, "answer": answer}


@app.post("/analyze")
async def analyze(request: Request):
    """B07 统一分析：5 工具 + ML 预测 → 证据 → 规则判断 → LLM 解释 → 存 analysis_runs。

    请求体：{"analysis_time": "2026-10-19 18:00"（可选，含 T 亦可）, "location": "南京"（可选）}
    缺省 analysis_time 时按当前时刻分析；LLM 解释失败不影响整体返回（explanation=null）。
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON"})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象"})

    location = (body.get("location") or "").strip() or analysis.DEFAULT_LOCATION

    scenario = (body.get("scenario") or "").strip().upper() or None
    if scenario and scenario not in analysis.SCENARIOS:
        return JSONResponse(
            status_code=400,
            content={"error": f"未知场景：{scenario}（支持：{'、'.join(analysis.SCENARIOS)}）"},
        )

    raw_time = (body.get("analysis_time") or "").strip()
    analysis_time = None
    if raw_time:
        normalized = raw_time.replace("T", " ")
        for fmt in (analysis.TS_FORMAT, "%Y-%m-%d %H:%M"):
            try:
                analysis_time = datetime.strptime(normalized, fmt)
                break
            except ValueError:
                continue
        if analysis_time is None:
            return JSONResponse(
                status_code=400,
                content={"error": "analysis_time 格式应为 YYYY-MM-DD HH:MM[:SS]"},
            )

    try:
        result = analysis.run_analysis(analysis_time, location, scenario)
    except Exception:
        logger.exception("Unexpected error handling /analyze")
        return JSONResponse(status_code=502, content={"error": "API调用失败：服务器内部错误"})
    logger.info(
        "Analyze succeeded (run_id=%s, time=%s, scenario=%s, evidence=%d, explanation=%s)",
        result["run_id"],
        result["analysis_time"],
        scenario or "S1",
        len(result["evidence"]),
        "yes" if result["explanation"] else "no",
    )
    return result


@app.get("/analyze/runs")
async def analyze_runs(limit: int = 20):
    """场景回放与对比：按时间倒序拉取 analysis_runs 历史（evidence/judgment 还原为对象）。"""
    limit = max(1, min(limit, 100))
    try:
        runs = db.latest_analysis_runs(limit)
    except Exception:
        logger.exception("Failed to list analysis runs")
        return JSONResponse(status_code=500, content={"error": "查询分析记录失败"})
    for run in runs:
        for key in ("evidence", "judgment"):
            try:
                run[key] = json.loads(run[key])
            except Exception:
                # 历史脏数据：保持原始字符串，前端按失败渲染
                logger.warning("Corrupted %s JSON in analysis run %s", key, run["run_id"])
    return {"count": len(runs), "runs": runs}


@app.get("/analyze/runs/{run_id}/export")
async def export_run(run_id: str, fmt: str = "md"):
    """导出单条 run 的完整证据链（attachment 下载）：fmt=md 可读报告 / fmt=json 原始记录。"""
    if fmt not in ("md", "json"):
        return JSONResponse(status_code=400, content={"error": "fmt 仅支持 md / json"})
    try:
        run = db.get_analysis_run(run_id)
    except Exception:
        logger.exception("Failed to fetch analysis run %s", run_id)
        return JSONResponse(status_code=500, content={"error": "查询分析记录失败"})
    if not run:
        return JSONResponse(status_code=404, content={"error": f"未找到 run_id：{run_id}"})
    short = run_id[:8]
    if fmt == "json":
        body = json.dumps(run, ensure_ascii=False, indent=2, default=str)
        return Response(
            content=body,
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="run_{short}.json"'},
        )
    body = analysis.run_to_markdown(run)
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="run_{short}_report.md"'},
    )
