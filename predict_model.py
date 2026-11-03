"""B06 模型推理：加载 attention_model.pkl，用最新数据预测 next_window_attention。

- load_model(path)：加载 pkl（含模型、特征列表、model_version、训练时间）。
- predict_attention(model_dict, rows)：输入最近 6 条环境数据（时间升序），
  用与训练完全一致的 8 个特征做预测，返回 predicted_attention 与 model_version。
- 命令行：python predict_model.py 直接取 SQLite 中最新 6 条数据演示。
"""
import argparse
import pickle
import statistics

import db
from ml_pipeline import FEATURES, WINDOW_POINTS

MODEL_PATH = "models/attention_model.pkl"


def load_model(path=MODEL_PATH):
    """加载 pkl，返回训练时保存的完整字典（model/features/model_version/...）。"""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_features(rows):
    """与 ml_pipeline.build_samples 完全一致的 8 特征计算（只用过去数据）。"""
    if len(rows) != WINDOW_POINTS:
        raise ValueError(f"需要最近 {WINDOW_POINTS} 条数据，实际 {len(rows)} 条")
    temps = [r["temperature"] for r in rows]
    humids = [r["humidity"] for r in rows]
    voltages = [r["voltage"] for r in rows]
    cur = rows[-1]
    values = {
        "temp_current": cur["temperature"],
        "temp_mean": statistics.mean(temps),
        "temp_delta": temps[-1] - temps[0],
        "temp_std": statistics.pstdev(temps),
        "humidity_current": cur["humidity"],
        "humidity_delta": humids[-1] - humids[0],
        "voltage_current": cur["voltage"],
        "voltage_mean": statistics.mean(voltages),
    }
    return [values[name] for name in FEATURES]  # 按训练时的特征顺序输出


def predict_attention(model_dict, rows):
    """对最新 rows 预测，返回 predicted_attention 与 model_version。"""
    X = build_features(rows)
    label = model_dict["model"].predict([X])[0]
    return {
        "predicted_attention": str(label),
        "model_version": model_dict.get("model_version", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(description="加载 attention_model.pkl，用 SQLite 最新数据预测")
    parser.add_argument("--model", default=MODEL_PATH, help="模型 pkl 路径")
    parser.add_argument("--n", type=int, default=WINDOW_POINTS, help="特征窗口点数（默认 6）")
    args = parser.parse_args()

    model_dict = load_model(args.model)
    # 完整性校验：pkl 里的特征列表必须与当前代码一致，否则预测结果不可信
    if model_dict.get("features") != FEATURES:
        print(f"警告：模型特征列表与当前代码不一致！\n  模型: {model_dict.get('features')}\n  代码: {FEATURES}")

    rows = db.latest_samples(args.n)
    result = predict_attention(model_dict, rows)

    print(f"模型: {args.model} | model_version: {result['model_version']}")
    print(f"数据窗口: {rows[0]['timestamp']} ~ {rows[-1]['timestamp']}（{len(rows)} 条，SIMULATED）")
    print(f"当前温度: {rows[-1]['temperature']} °C | 湿度: {rows[-1]['humidity']} % | 电压: {rows[-1]['voltage']} V")
    print(f"特征: {dict(zip(FEATURES, [round(v, 3) for v in build_features(rows)]))}")
    print(f"预测结果: predicted_attention = {result['predicted_attention']}")


if __name__ == "__main__":
    main()
