"""B06 传统机器学习闭环：数据 → 滑动窗口特征 → 未来标签 → 时间切分。

数据：environment_samples（B02 生成的 336 条 SIMULATED 数据，30 分钟间隔）。
特征：用最近 6 个点（3 小时）提取 8 个特征——只允许使用过去数据；未来数据
      仅用于打标签。表内没有 sim_scenario 列，字段一律按白名单（timestamp +
      温度/湿度/电压）提取，场景字段不可能混入特征。
标签：看未来 4 个点（2 小时）的温度变化，任一时刻与当前温度之差的绝对值
      >= 阈值 → next_window_attention=WATCH，否则 NORMAL。
切分：前 5 天训练、后 2 天测试；窗口（特征 6 点 + 标签 4 点）跨边界的直接
      丢弃，保证训练集绝不包含测试集时间的数据。
"""
import argparse
import csv
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

import db

WINDOW_POINTS = 6  # 特征窗口：最近 6 个点 = 3 小时
LOOKAHEAD_POINTS = 4  # 标签窗口：未来 4 个点 = 2 小时
# WATCH 判定阈值（°C）。数据实测：5 个扰动事件（9~13°C）随机全部落在前 5 天，
# 后 2 天只有纯昼夜变化（2 小时最大偏离约 3.96°C）。阈值 >= 4.0 时测试集
# WATCH=0（无法评估召回），3.5 时训练/测试都有正样本（66/13），故取 3.5。
WATCH_THRESHOLD = 3.5
TRAIN_DAYS = 5  # 前 5 天训练，后 2 天测试
DEFAULT_OUT_DIR = "ml_data"

# 8 个特征：温度 4 个 + 湿度 2 个 + 电压 2 个（全部来自过去 6 个点）
FEATURES = [
    "temp_current",  # 当前温度
    "temp_mean",  # 窗口平均温度
    "temp_delta",  # 温度变化量 delta（窗口末值 - 首值，3 小时变化）
    "temp_std",  # 窗口温度标准差（波动程度）
    "humidity_current",  # 当前湿度
    "humidity_delta",  # 窗口湿度变化量
    "voltage_current",  # 当前电压
    "voltage_mean",  # 窗口平均电压
]

TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def load_rows():
    """读取全部历史数据并按时间排序（query_history 本身按 timestamp ASC）。"""
    rows = db.query_history(None, None, 5000)
    parsed = []
    for r in rows:
        # 白名单提取：只保留时间戳 + 三个信号列，其余列（如 source_type）
        # 一律不进入后续流程，未来也不可能有 sim_scenario 之类的字段混入
        parsed.append(
            {
                "timestamp": r["timestamp"],
                "ts": datetime.strptime(r["timestamp"], TS_FORMAT),
                "temperature": r["temperature"],
                "humidity": r["humidity"],
                "voltage": r["voltage"],
            }
        )
    return parsed


def build_samples(rows, watch_threshold):
    """滑动窗口：特征只看过去（i-5..i），标签只看未来（i+1..i+4）。"""
    samples = []
    n = len(rows)
    for i in range(WINDOW_POINTS - 1, n - LOOKAHEAD_POINTS):
        window = rows[i - WINDOW_POINTS + 1 : i + 1]  # 过去 6 个点（含当前）
        current = rows[i]
        future = rows[i + 1 : i + LOOKAHEAD_POINTS + 1]  # 未来 4 个点

        temps = [w["temperature"] for w in window]
        humids = [w["humidity"] for w in window]
        voltages = [w["voltage"] for w in window]

        features = {
            "temp_current": round(current["temperature"], 3),
            "temp_mean": round(statistics.mean(temps), 3),
            "temp_delta": round(temps[-1] - temps[0], 3),
            "temp_std": round(statistics.pstdev(temps), 3),
            "humidity_current": round(current["humidity"], 3),
            "humidity_delta": round(humids[-1] - humids[0], 3),
            "voltage_current": round(current["voltage"], 3),
            "voltage_mean": round(statistics.mean(voltages), 3),
        }

        # 标签：未来 2 小时内任一时刻温度相对当前急剧上升/下降 → WATCH
        max_dev = max(abs(f["temperature"] - current["temperature"]) for f in future)
        label = "WATCH" if max_dev >= watch_threshold else "NORMAL"

        samples.append(
            {
                "timestamp": current["timestamp"],
                **features,
                "next_window_attention": label,
                # 内部字段：窗口首点（特征）与标签末点的时间，用于切分，不写进 CSV
                "_window_start": window[0]["ts"],
                "_future_end": future[-1]["ts"],
            }
        )
    return samples


def split_samples(samples, boundary):
    """前 TRAIN_DAYS 天训练、其余测试；整个窗口跨边界的一律丢弃。"""
    train, test, dropped = [], [], 0
    for s in samples:
        if s["_future_end"] <= boundary:
            train.append(s)  # 标签末点都不超过边界 → 完全在训练时间段内
        elif s["_window_start"] > boundary:
            test.append(s)  # 特征首点严格晚于边界 → 完全在测试时间段内
        else:
            dropped += 1  # 窗口跨越边界，丢弃
    return train, test, dropped


def write_csv(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp"] + FEATURES + ["next_window_attention"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig 便于 Excel 打开
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for s in samples:
            writer.writerow({k: s[k] for k in header})
    return len(samples)


def dist(samples):
    counts = {"NORMAL": 0, "WATCH": 0}
    for s in samples:
        counts[s["next_window_attention"]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="B06 滑动窗口特征工程 + 未来标签 + 时间切分")
    parser.add_argument("--window-points", type=int, default=WINDOW_POINTS, help="特征窗口点数（默认 6 = 3 小时）")
    parser.add_argument("--lookahead-points", type=int, default=LOOKAHEAD_POINTS, help="标签窗口点数（默认 4 = 2 小时）")
    parser.add_argument("--watch-threshold", type=float, default=WATCH_THRESHOLD, help="WATCH 判定阈值 °C（默认 3.5）")
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS, help="训练天数（默认前 5 天）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="CSV 输出目录（默认 ml_data）")
    args = parser.parse_args()

    rows = load_rows()
    if len(rows) < args.window_points + args.lookahead_points:
        print(f"数据不足：需要至少 {args.window_points + args.lookahead_points} 行，实际 {len(rows)} 行")
        sys.exit(1)

    boundary = rows[0]["ts"] + timedelta(days=args.train_days)

    samples = build_samples(rows, args.watch_threshold)
    train, test, dropped = split_samples(samples, boundary)

    out_dir = Path(args.out_dir)
    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    write_csv(train_csv, train)
    write_csv(test_csv, test)

    train_dist = dist(train)
    test_dist = dist(test)

    print("=" * 60)
    print("B06 特征工程与切分结果")
    print("=" * 60)
    print(f"数据源: {db.DB_PATH.name} | {len(rows)} 行")
    print(f"时间范围: {rows[0]['timestamp']} ~ {rows[-1]['timestamp']}")
    print(f"特征窗口: 过去 {args.window_points} 点 | 标签窗口: 未来 {args.lookahead_points} 点")
    print(f"WATCH 阈值: 未来 2 小时内温度偏离当前 >= {args.watch_threshold} °C")
    print(f"切分边界: 前 {args.train_days} 天 <= {boundary.strftime(TS_FORMAT)} < 测试集")
    print(f"有效样本: {len(samples)}（跨边界丢弃 {dropped} 个窗口）")
    print(f"训练集: {len(train)} 行 | 标签分布 NORMAL={train_dist['NORMAL']}, WATCH={train_dist['WATCH']}")
    print(f"测试集: {len(test)} 行 | 标签分布 NORMAL={test_dist['NORMAL']}, WATCH={test_dist['WATCH']}")
    print(f"特征（{len(FEATURES)} 个）: {', '.join(FEATURES)}")
    print(f"已写出: {train_csv} ({len(train)} 行), {test_csv} ({len(test)} 行)")


if __name__ == "__main__":
    main()
