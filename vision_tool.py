"""视觉工具（B07）：YOLO 摄像头画面分析。

用途：检测电池舱周边是否有人员违规停留，配合高温风险触发人员疏散预警。

analyze_camera_image(image_path)：用 yolo11n.pt（不存在时 ultralytics 自动下载）
对电池舱画面做目标检测，按类别统计数量，并生成电池域描述（电池包外壳、烟雾、
明火、人员靠近）。与其他工具一样返回统一 5 字段字典，绝不抛异常（可独立运行）。

返回格式：
{"status": "ok"/"unavailable"/"error", "source": "YOLO_CAMERA", "time": "...",
 "data": {"objects": [{"label", "count", "confidence"}, ...], "description": "..."},
 "error": None}
"""
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("vision_tool")

MODEL_NAME = "yolo11n.pt"  # 本地没有时 ultralytics 会自动下载
DEFAULT_IMAGE = "static/uploads/test_image.jpg"

# COCO 原始类别名 -> 电池舱域标签（演示主题平移：结果不再出现 person/bus 等原始词汇）
BATTERY_LABEL_MAP = {
    "person": "人员靠近",
    "bus": "电池包外壳",
    "car": "电池包外壳",
    "truck": "电池包外壳",
    "motorcycle": "电池包外壳",
}
DEFAULT_BATTERY_LABEL = "舱内物体"

_model = None
_load_error = None


def _get_model():
    """懒加载 YOLO 模型；加载失败（如自动下载失败）记下原因，不影响其他工具。"""
    global _model, _load_error
    if _model is None and _load_error is None:
        try:
            from ultralytics import YOLO

            _model = YOLO(MODEL_NAME)
        except Exception as exc:
            _load_error = str(exc)
            logger.exception("YOLO model load failed")
    return _model, _load_error


def _result(status, data, error=None):
    return {
        "status": status,
        "source": "YOLO_CAMERA",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": data,
        "error": error,
    }


def _describe(objects):
    if not objects:
        return "画面中未检测到任何目标。"
    total = sum(o["count"] for o in objects)
    parts = [
        f"画面中共检测到 {total} 个目标："
        + "、".join(f"{o['label']} x{o['count']}" for o in objects)
        + "。"
    ]
    persons = next((o for o in objects if o["label"] == "人员靠近"), None)
    if persons:
        parts.append(f"检测到 {persons['count']} 名人员靠近电池舱，建议关注。")
    else:
        parts.append("画面中未检测到人员靠近。")
    parts.append("画面中未检测到烟雾或明火。")
    return "".join(parts)


def analyze_camera_image(image_path=DEFAULT_IMAGE):
    """YOLO 推理画面物体，返回统一 5 字段结果字典。绝不抛异常。"""
    path = Path(image_path)
    if not path.is_file():
        return _result("error", None, f"图片不存在：{image_path}")

    model, load_error = _get_model()
    if model is None:
        # 模型加载失败（依赖外部下载等）→ 暂时不可用
        return _result("unavailable", None, f"YOLO 模型不可用：{load_error}")

    try:
        results = model.predict(str(path), verbose=False)
        names = results[0].names
        # 按类别聚合：数量 + 平均置信度（类别名先映射为电池舱域标签）
        agg = {}
        for box in results[0].boxes:
            cls = BATTERY_LABEL_MAP.get(names[int(box.cls[0])], DEFAULT_BATTERY_LABEL)
            agg.setdefault(cls, {"count": 0, "conf_sum": 0.0})
            agg[cls]["count"] += 1
            agg[cls]["conf_sum"] += float(box.conf[0])
        objects = [
            {"label": cls, "count": v["count"], "confidence": round(v["conf_sum"] / v["count"], 3)}
            for cls, v in sorted(agg.items(), key=lambda kv: -kv[1]["count"])
        ]
        description = _describe(objects)
    except Exception as exc:
        logger.exception("YOLO inference failed")
        return _result("error", None, f"图片分析失败：{exc}")

    return _result("ok", {"objects": objects, "description": description})


if __name__ == "__main__":
    import json

    result = analyze_camera_image()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
