# GreenMate 版本记录（Version Record）

| 项 | 值 |
|---|---|
| app_version | B08_baseline_01 |
| model_version | rf_v1 |
| git tag | v1.0-final |
| 冻结日期 | 2026-10-21 |

## 本次冻结内容

- **scenarios.json**：S1~S5 固定注入配方（当前环境数值 / 48 点历史曲线 / 24h 天气数值 / 天气与 YOLO 状态 / S2 的 ML 规则覆盖），/analyze 运行时严格读取，禁止随机生成，保证每次运行结果完全一致
- 前端场景下拉框（S1 正常基线 / S2 渐进漂移 / S3 外部扰动 / S4 证据分歧 / S5 证据缺失）
- 场景回放与对比（结构化差异表）+ 导出运行报告（md/json）
- S2 漂移规则：24h 升温 Δ≥5.0°C 时后端规则将关注等级提升为 WATCH（结构化判断带 basis）
- 状态演化事件标注（NORMAL→WATCH 转折 / 外部天气条件介入）

## 模型与边界

- models/attention_model.pkl：rf_v1（RandomForest，8 特征，6 点滑窗）
- 产品边界：环境认知（不涉及病害诊断）
