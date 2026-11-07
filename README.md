# ThermoGuard 电池热失控预警系统 (v2.0-thermoguard-final)

## 复现性声明

本项目所有测试结果均完全可复现。系统已通过锁定 scenarios.json 和模型版本（rf_v1），消除了运行时的随机性。任何评委按照以下指南，均可得到与验证记录中完全一致的 S1~S5 结构化判断结果。

本项目为面向锂电池安全监测场景的轻量级状态认知与多源证据分析系统。包含 RAG 防幻觉、Agent 工具调用、传统机器学习预测及 YOLO 视觉感知。

## 环境要求

- Python 3.10+
- 可联网环境（用于调用大模型 API）

## 快速复现指南（10 分钟）

### 1. 安装依赖

在根目录打开终端，创建并激活虚拟环境：

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Mac/Linux

pip install -r requirements.txt
```

### 2. 配置密钥

```bash
copy .env.example .env     # Windows
# cp .env.example .env     # Mac/Linux
```

打开 .env 文件，填入你的大模型 API Key（如 DeepSeek）。

### 3. 首次启动与数据初始化（关键）

在启动 Web 服务前，必须先运行脚本生成历史数据（336 条云端下采样环境样本）：

> 采样策略：云端正向降采样（Downsampling）——车端硬件采用高频采集，云端分析采用 30 分钟下采样，以提取宏观长周期趋势特征，避免高频噪声干扰模型。

```bash
python simulate_data.py
```

### 4. 启动 Web 服务

```bash
uvicorn main:app --reload --port 8080
```

在浏览器中访问：http://127.0.0.1:8080

### 5. 随包文件说明

- 历史数据库：运行 simulate_data.py 后自动在当前目录生成 sensor.db。
- ML 模型：已随包提供，位于 models/attention_model.pkl。
- 测试图片：已随包提供，位于 static/uploads/test_image.jpg（用于 YOLO 视觉验证）。
- YOLO 权重：如项目根目录无 yolo11n.pt，首次调用视觉工具时系统会自动从网络下载。

## 核心功能验证流程

### 1. Agent 验证

在主控台输入“查一下明天的天气”，观察 AI 是否反问城市；输入“苏州”，查看终端是否打印真实的 [Tool Call]。

### 2. RAG 验证

询问“电池怎么消毒”，确认系统回答“资料不足”。

### 3. S5 降级验证

1. 删除或重命名 static/uploads/test_image.jpg。
2. 在网页下拉框选择 S5 并运行。
3. 观察证据完整度降至 4/6，且 AI 输出“判断受限”。
