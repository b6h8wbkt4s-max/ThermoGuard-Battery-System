"""生成连续多天的模拟电池舱环境数据并写入 SQLite。

数据特征：
- 正常昼夜变化：电池温度 14 时最高、凌晨最低，整体处于电池包正常工作温度 20~45°C 区间；
  电池包内湿度与温度反相；电池电压围绕 3.7V 标称值微幅波动，
  并始终钳制在单体电压合理区间 2.5V ~ 4.2V 内
- 随机插入 4-6 个连续扰动事件（温度突升、湿度下降，持续数小时后恢复）
- 所有数据 source_type = SIMULATED
- 采样策略：云端正向降采样（Downsampling）——车端硬件采用高频采集，
  云端分析采用 30 分钟下采样，以提取宏观长周期趋势特征，避免高频噪声干扰模型

重复执行会先清空旧的 SIMULATED 数据再写入（幂等）。

用法：python simulate_data.py [--days 7] [--interval-minutes 30] [--seed 42]
"""
import argparse
import math
import random
from datetime import datetime, timedelta

import db

SOURCE_TYPE = "SIMULATED"


def generate(days, interval_minutes, seed):
    rng = random.Random(seed)
    step = timedelta(minutes=interval_minutes)  # 云端正向降采样步长（Downsampling，30min/点）
    end = datetime.now().replace(second=0, microsecond=0)
    # 对齐到整点间隔，保证时间戳规整（如 08:00 / 08:30）
    end -= timedelta(minutes=end.minute % interval_minutes)
    start = end - timedelta(days=days) + step
    total = int(days * 24 * 60 / interval_minutes)

    # 基线昼夜曲线
    temps, hums, voltages = [], [], []
    for i in range(total):
        t = start + i * step
        hour = t.hour + t.minute / 60
        # 电池温度：14 时最高 ~36°C，凌晨 2 时最低 ~24°C，处于正常工作温度 20~45°C 区间，加小幅噪声
        temp = 30 + 6 * math.cos(2 * math.pi * (hour - 14) / 24) + rng.uniform(-0.3, 0.3)
        # 电池包内湿度：与温度反相，白天 ~37%，夜间 ~53%
        hum = 45 - 8 * math.cos(2 * math.pi * (hour - 14) / 24) + rng.uniform(-1.5, 1.5)
        # 电池电压：围绕 3.7V 标称值微幅波动（±0.02V 昼夜纹波 + 小噪声），钳制在 2.5~4.2V 合理区间
        voltage = 3.7 + 0.02 * math.cos(2 * math.pi * (hour - 14) / 24) + rng.uniform(-0.005, 0.005)
        voltage = min(4.2, max(2.5, voltage))
        temps.append(temp)
        hums.append(hum)
        voltages.append(voltage)

    # 随机扰动事件（温升事件）：4-6 个，互不重叠；梯形窗口（前后约 1 小时过渡）
    occupied = []
    events = []
    event_count = rng.randint(4, 6)
    attempts = 0
    while len(events) < event_count and attempts < 200:
        attempts += 1
        duration_points = max(3, int(rng.uniform(3, 8) * 60 / interval_minutes))
        start_i = rng.randrange(total - duration_points)
        end_i = start_i + duration_points
        if any(start_i < o_end and end_i > o_start for (o_start, o_end) in occupied):
            continue  # 与已有事件重叠，重新采样
        occupied.append((start_i, end_i))
        d_temp = rng.uniform(2, 8)   # 温升事件幅度控制在正常区间上限以内
        d_hum = rng.uniform(-8, -3)
        ramp = 2  # 前后各 2 个点（约 1 小时）线性过渡
        for j in range(start_i, end_i):
            if j - start_i < ramp:
                w = (j - start_i + 1) / ramp
            elif end_i - 1 - j < ramp:
                w = (end_i - j) / ramp
            else:
                w = 1.0
            temps[j] += d_temp * w
            hums[j] += d_hum * w
        events.append((start_i, end_i, d_temp, d_hum))

    rows = []
    for i in range(total):
        t = start + i * step
        rows.append((
            t.strftime("%Y-%m-%d %H:%M:%S"),
            round(temps[i], 2),
            round(max(10.0, min(95.0, hums[i])), 2),
            round(voltages[i], 2),
            SOURCE_TYPE,
        ))
    return rows, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="生成天数，默认 7")
    parser.add_argument("--interval-minutes", type=int, default=30, help="云端降采样间隔（分钟），默认 30（Downsampling 策略）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42（可复现）")
    args = parser.parse_args()

    db.init_db()
    rows, events = generate(args.days, args.interval_minutes, args.seed)
    db.replace_simulated(rows)
    print(f"已写入 {len(rows)} 条 SIMULATED 数据到 {db.DB_PATH}")
    print(f"时间范围: {rows[0][0]} ~ {rows[-1][0]}")
    print(f"扰动事件: {len(events)} 个")
    for i, (s, e, d_temp, d_hum) in enumerate(events, 1):
        print(f"  #{i} {rows[s][0]} ~ {rows[e - 1][0]}  温度 +{d_temp:.1f}°C  湿度 {d_hum:.1f}%")


if __name__ == "__main__":
    main()
