"""B06 模型训练与评估：随机森林 + 朴素基线对比 + 模型保存。

输入：ml_pipeline.py 生成的 ml_data/train.csv / test.csv（前 5 天训练、后 2 天
测试，跨边界窗口已丢弃，无时间泄漏）。
输出：测试集混淆矩阵、WATCH 类 F1/Recall、朴素基线（全猜 NORMAL）对比；
模型保存为 models/rf_watch.pkl（含特征列表与训练时间），特征列表另存
models/features.json。
"""
import argparse
import csv
import json
import pickle
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from ml_pipeline import FEATURES  # 复用同一份特征定义，训练/推理特征保持一致

LABEL_COL = "next_window_attention"
DEFAULT_DATA_DIR = "ml_data"
DEFAULT_MODEL_DIR = "models"
MODEL_VERSION = "rf_v1"  # 随模型结构/特征/训练数据变化时递增

RF_PARAMS = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}


def load_csv(path):
    """读取 ml_pipeline 输出的 CSV，返回 (X, y, timestamps)。"""
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    X = [[float(r[f]) for f in FEATURES] for r in rows]
    y = [r[LABEL_COL] for r in rows]
    timestamps = [r["timestamp"] for r in rows]
    return X, y, timestamps


def print_confusion_matrix(y_true, y_pred):
    """混淆矩阵：行=真实标签，列=预测标签。"""
    labels = ["NORMAL", "WATCH"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("混淆矩阵（行=真实，列=预测）:")
    header = " " * 14 + "预测NORMAL" + " " * 3 + "预测WATCH"
    print(header)
    for i, label in enumerate(labels):
        row = f"真实{label:<11}" + f"{cm[i][0]:>12}" + f"{cm[i][1]:>12}"
        print(row)


def baseline_metrics(y_true):
    """朴素基线：全部预测 NORMAL（多数类）。"""
    y_pred = ["NORMAL"] * len(y_true)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "watch_recall": recall_score(y_true, y_pred, pos_label="WATCH", zero_division=0),
        "watch_f1": f1_score(y_true, y_pred, pos_label="WATCH", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="B06 随机森林训练与评估")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="CSV 数据目录（默认 ml_data）")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型输出目录（默认 models）")
    parser.add_argument("--n-estimators", type=int, default=RF_PARAMS["n_estimators"], help="随机森林树数量")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X_train, y_train, _ = load_csv(data_dir / "train.csv")
    X_test, y_test, _ = load_csv(data_dir / "test.csv")
    print(f"训练集 {len(y_train)} 行（WATCH={y_train.count('WATCH')}），"
          f"测试集 {len(y_test)} 行（WATCH={y_test.count('WATCH')}）")

    # 朴素基线：全猜 NORMAL
    base = baseline_metrics(y_test)
    print("\n== 朴素基线（全猜 NORMAL）==")
    print(f"accuracy={base['accuracy']:.3f}, WATCH recall={base['watch_recall']:.3f}, "
          f"WATCH F1={base['watch_f1']:.3f}")

    # 随机森林
    params = {**RF_PARAMS, "n_estimators": args.n_estimators}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n== 随机森林（测试集）==")
    print_confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred, labels=["NORMAL", "WATCH"], digits=3))

    print("== 对比 ==")
    print(f"{'':<12}{'accuracy':>10}{'WATCH召回':>12}{'WATCH F1':>12}")
    print(f"{'朴素基线':<12}{base['accuracy']:>10.3f}{base['watch_recall']:>12.3f}{base['watch_f1']:>12.3f}")
    print(f"{'随机森林':<12}{accuracy_score(y_test, y_pred):>10.3f}"
          f"{recall_score(y_test, y_pred, pos_label='WATCH'):>12.3f}"
          f"{f1_score(y_test, y_pred, pos_label='WATCH'):>12.3f}")

    # 保存模型 + 特征列表
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = model_dir / "attention_model.pkl"
    features_path = model_dir / "features.json"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": FEATURES,
                "label_column": LABEL_COL,
                "params": params,
                "model_version": MODEL_VERSION,
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
        )
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "features": FEATURES,
                "label_column": LABEL_COL,
                "params": params,
                "model_version": MODEL_VERSION,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n已保存模型: {pkl_path}（含特征列表与训练时间）")
    print(f"已保存特征列表: {features_path}")


if __name__ == "__main__":
    main()
