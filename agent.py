"""LLM Tool Agent（B05/B06/B07）。

- TOOL_DEFINITIONS：5 个工具的 OpenAI function calling 格式定义（4 个数据工具
  来自 tools.py，1 个视觉工具来自 vision_tool.py）；
- run_agent()：工具调用闭环——问题发给 LLM（附带 tools）→ LLM 返回 tool_calls
  → json.loads 解析函数名与参数 → Python 端真实执行 → 结果打包为 role: tool
  消息回传 → LLM 生成最终回答。最多执行 3 次工具，防止死循环。

Description 约定：
- 写清每个工具的用途与数据来源（SIMULATED / PUBLIC / KNOWLEDGE_BASE）；
- 查天气必须提供 location 参数：用户未说明地点时先询问，不得猜测。
"""
import json
import logging

import tools
import vision_tool
from llm_client import chat
from tools import _result  # 统一的 5 字段结果结构，合成错误结果时保持一致

logger = logging.getLogger("agent")

MAX_TOOL_CALLS = 3  # 最多执行工具的次数，防止死循环

SYSTEM_PROMPT = (
    "你是一个锂电池储能安全监测助手，可以通过调用工具查询真实数据。\n"
    "工具选择规则（按用户问题判断，必须遵守）：\n"
    "- 用户询问天气/天气预报/降水概率 → 必须调用 get_weather_forecast，location 为用户所说的地点；\n"
    "- 用户询问历史数据/变化趋势/过去某段时间 → 调用 get_environment_history；\n"
    "- 用户询问当前/现在的电池舱环境 → 调用 get_current_environment；\n"
    "- 用户询问电池安全/热失控前兆/BMS 管理知识 → 调用 search_battery_knowledge；\n"
    "- 用户询问画面/摄像头/高压危险区人员违规停留 → 调用 analyze_camera_image；\n"
    "其他规则：\n"
    "1. 与数据相关的问题必须先调用对应工具获取真实数据，不得凭空编造数值；\n"
    "2. 查询天气时用户没有说明地点：先向用户询问地点，不要猜测，也不要用其他工具代替回答；\n"
    "3. 根据工具返回的 status 组织回答：ok 使用 data；unavailable 告知数据暂时不可用及原因；error 告知查询失败及原因；\n"
    "4. 回答时说明数据来源：SIMULATED 为本地模拟数据、PUBLIC 为外部公开数据、KNOWLEDGE_BASE 为本地知识库；\n"
    "5. 只回答用户问的内容，不要用无关数据代替；\n"
    "6. 如果有人问及数据频率，必须回答：本系统为云端离线复盘，非车端实时控制。"
)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_environment",
            "description": (
                "获取电池舱当前环境的实时模拟状态，包含电池温度、电池包内湿度、电池电压。"
                "数据来源为本地模拟数据（SIMULATED），无需任何参数。"
                "当用户询问「电池舱现在/当前的环境怎么样」时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_environment_history",
            "description": (
                "查询最近若干小时的电池舱环境历史数据（每 30 分钟一个采样点），"
                "包含电池温度、电池包内湿度、电池电压。数据来源为本地 SQLite 数据库（SIMULATED）。"
                "当用户询问电池温度变化趋势或过去某段时间的数据时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "向前查询的小时数，必须为正数；不填则默认 24 小时。",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": (
                "获取指定地点未来 3 天的逐小时天气预报（温度、湿度、降水概率）。"
                "数据来自 Open-Meteo（PUBLIC 外部数据）。"
                "调用时必须提供 location 参数：如果用户没有说明地点，"
                "必须先向用户询问地点，不得猜测、省略或使用默认值。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "要查询天气的城市名，任意城市均可（中文城市名，如“北京”“上海”）。"
                            "若用户未说明地点，不要调用本工具，先向用户询问地点。"
                        ),
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_battery_knowledge",
            "description": (
                "在本地锂电池安全知识库中检索与问题最相关的 2 个段落。"
                "知识库内容为锂电池热失控前兆、温度阈值与 BMS 管理规范（KNOWLEDGE_BASE）。"
                "当用户询问电池安全/BMS 管理相关专业知识时使用；检索结果需结合参考资料回答，"
                "知识库中没有的内容要明确告知资料不足，不得编造。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索词或问题，例如：锂电池正常工作温度范围是多少。",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_camera_image",
            "description": (
                "用 YOLO 分析电池舱摄像头画面，用于检测电池舱周边是否有人员违规停留，"
                "配合高温风险触发人员疏散预警；同时识别画面中的目标（电池包外壳、烟雾、明火、人员靠近等）"
                "并统计数量。数据来源为本地摄像头画面（YOLO_CAMERA）。"
                "当用户询问「画面里有什么」「高压危险区是否有人员违规停留」「摄像头看到什么」等问题时调用。"
                "image_path 不填时使用默认测试画面 static/uploads/test_image.jpg。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "图片文件路径；不填则分析默认画面 static/uploads/test_image.jpg。",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
]

# 工具名 -> tools.py 中对应的真实函数
TOOL_DISPATCH = {
    "get_current_environment": tools.get_current_environment,
    "get_environment_history": tools.get_environment_history,
    "get_weather_forecast": tools.get_weather_forecast,
    "search_battery_knowledge": tools.search_battery_knowledge,
    "analyze_camera_image": vision_tool.analyze_camera_image,
}


def definitions_for_model():
    """返回可直接放入 chat completions tools 参数的列表。"""
    return TOOL_DEFINITIONS


def _execute(tool_call):
    """解析并执行单个工具调用，返回统一的 5 字段结果字典。绝不抛异常。"""
    name = tool_call["name"]
    raw = tool_call["arguments"] or "{}"
    try:
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError("arguments 必须是 JSON 对象")
    except Exception as exc:
        print(f"[Tool Call] 调用了 {name}，参数：{raw}（解析失败：{exc}）", flush=True)
        result = _result("error", "AGENT", None, f"工具参数解析失败：{exc}")
        print(f"[Tool Result] 状态：{result['status']}，错误：{result['error']}", flush=True)
        return result

    print(f"[Tool Call] 调用了 {name}，参数：{args}", flush=True)
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        result = _result("error", "AGENT", None, f"未知工具：{name}")
    else:
        try:
            result = fn(**args)
        except TypeError as exc:
            # 模型给出的参数与实际签名不匹配：转成统一错误，不让闭环崩溃
            result = _result("error", "AGENT", None, f"工具参数不匹配：{exc}")
        except Exception as exc:
            logger.exception("Tool %s raised unexpected error", name)
            result = _result("error", "AGENT", None, f"工具执行异常：{exc}")
    print(
        f"[Tool Result] 状态：{result['status']}"
        + (f"，错误：{result['error']}" if result.get("error") else ""),
        flush=True,
    )
    return result


def run_agent(question):
    """工具调用闭环。返回 (最终回答, 实际执行的工具次数)。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    executed = 0
    for _ in range(MAX_TOOL_CALLS):
        message = chat(messages, tools=definitions_for_model())
        tool_calls = message["tool_calls"]
        if not tool_calls:
            return message["content"], executed
        # 把 assistant 消息（含 tool_calls）原样记入上下文，OpenAI 格式要求回带
        messages.append(
            {
                "role": "assistant",
                "content": message["content"],
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            if executed >= MAX_TOOL_CALLS:
                print(
                    f"[Tool Call] 已达到最大执行次数（{MAX_TOOL_CALLS}），拒绝执行 {tc['name']}",
                    flush=True,
                )
                result = _result(
                    "error", "AGENT", None, f"已达到最大工具执行次数（{MAX_TOOL_CALLS}），本次调用被拒绝"
                )
            else:
                result = _execute(tc)
                executed += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )
    # 用满 3 次仍要工具：强制模型基于已有工具结果直接回答，不再提供工具
    messages.append(
        {
            "role": "user",
            "content": "已达到最大工具调用次数，请根据以上工具执行结果直接给出最终回答，不要再调用工具。",
        }
    )
    final = chat(messages, tools=None)
    return final["content"], executed


if __name__ == "__main__":
    # 校验：每个工具定义都对应 TOOL_DISPATCH 中真实可调用的函数
    print("== 校验工具定义与 TOOL_DISPATCH 函数一一对应 ==")
    ok = True
    for tool in TOOL_DEFINITIONS:
        name = tool["function"]["name"]
        if callable(TOOL_DISPATCH.get(name)):
            print(f"[OK] {name}")
        else:
            print(f"[MISSING] {name} 不在 TOOL_DISPATCH 中")
            ok = False
    print(f"\n共 {len(TOOL_DEFINITIONS)} 个工具定义，校验{'通过' if ok else '失败'}")
    print("\n== TOOL_DEFINITIONS ==")
    print(json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2))
