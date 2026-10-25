"""环境状态：模拟数据源、当前状态与 Prompt 拼装（被 main.py 与 tools.py 共用，避免循环导入）。"""

FIXED_QUESTION = "电池舱当前环境怎么样？"

# 两组模拟环境数据（电池舱）
# 电池电压：单体电压合理区间 2.5V ~ 4.2V（标称 3.7V；高温预警时电压小幅下探至 3.2V）
STATES = {
    "A": {"name": "正常运行", "temperature": 30, "humidity": 50, "voltage": 3.7},
    "B": {"name": "高温预警", "temperature": 58, "humidity": 30, "voltage": 3.2},
}

_current_key = "A"


def get_current_key():
    return _current_key


def get_current_state():
    return STATES[_current_key]


def set_current_state(key):
    global _current_key
    _current_key = key
    return STATES[key]


def toggle_state():
    """切换到另一组状态，返回切换后的 key。"""
    set_current_state("B" if _current_key == "A" else "A")
    return _current_key


def state_data(state_key):
    next_key = "B" if state_key == "A" else "A"
    return {"source_type": "SIMULATED", **STATES[state_key], "next_name": STATES[next_key]["name"]}


def build_prompt(state):
    return (
        "请根据以下电池舱当前环境数据回答用户的问题。\n"
        f"当前环境状态：\n"
        f"- 电池温度：{state['temperature']} °C\n"
        f"- 电池包内湿度：{state['humidity']} %\n"
        f"- 电池电压：{state['voltage']} V\n"
        f"\n用户问题：{FIXED_QUESTION}"
    )
