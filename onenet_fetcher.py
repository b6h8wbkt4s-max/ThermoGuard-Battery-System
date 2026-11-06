"""OneNET 云端数据拉取器：定时从 OneNET Studio 拉取设备最新数据并存入本地 sensor.db。

流程：每隔 3 秒调用一次 OneNET Studio 物模型属性查询接口（iot-api.heclouds.com），
取产品 ONENET_PRODUCT_ID / 设备 ONENET_DEVICE_NAME 最新上报的 temperature / voltage / humidity；
先推送到本地 FastAPI 的 /onenet/webhook 入库（缺省 http://127.0.0.1:8080），
webhook 不可达时回退为直接写入 sensor.db，保证拉取成功的数据不丢。

鉴权：读取 .env 中的产品级 ONENET_ACCESS_KEY（OneNET Studio 控制台「安全鉴权」页生成），
按官方安全鉴权算法实时生成 token：signStr = et\\nmethod\\nres\\nversion，
sign = base64(HMAC-SHA1(accessKey, signStr))，version 固定 2020-05-29，
产品级 AccessKey 对应 res = products/{product_id}；未配置时脚本拒绝启动，绝不用模拟数据冒充云数据。

用法：
  python onenet_fetcher.py
  python onenet_fetcher.py --interval 3 --webhook http://127.0.0.1:8080/onenet/webhook
"""
import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 与 llm_client.py 一致：load_dotenv 默认只找当前工作目录，换目录启动会导致读不到 Key
load_dotenv(Path(__file__).resolve().parent / ".env")

# Windows 控制台默认 GBK，避免中文日志乱码；line_buffering 保证日志逐行实时刷新到终端
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] onenet_fetcher: %(message)s",
)
logger = logging.getLogger("onenet_fetcher")

PRODUCT_ID = (os.getenv("ONENET_PRODUCT_ID") or "").strip()
DEVICE_NAME = (os.getenv("ONENET_DEVICE_NAME") or "").strip()

# 物模型属性标识符，与本地 /onenet/webhook 的字段一一对应
PROPERTY_KEYS = ("temperature", "voltage", "humidity")

# OneNET Studio 物模型属性查询接口（IoT Open API），可用环境变量 ONENET_API_URL 覆盖
ONENET_API = os.getenv(
    "ONENET_API_URL",
    "https://iot-api.heclouds.com/thingmodel/query-device-property",
)

DEFAULT_WEBHOOK = "http://127.0.0.1:8080/onenet/webhook"
HTTP_TIMEOUT = 5  # 云端请求与本地推送的单个请求超时（秒）

# OneNET Studio 安全鉴权固定版本号；token 有效期 1 小时，每次请求现算，无需续期
AUTH_VERSION = "2020-05-29"
TOKEN_TTL = 3600


def build_auth_token(access_key, product_id):
    """按 OneNET Studio 官方安全鉴权算法（HMAC-SHA1）生成产品级 token。

    签名串：et\\nmethod\\nres\\nversion；产品级 AccessKey 对应资源 res = products/{product_id}；
    控制台生成的 accessKey 是 base64 编码的密钥，参与 HMAC 前必须先 b64decode 为原始字节；
    sign = base64(HMAC-SHA1(key_bytes, signStr))；res 与 sign 均需 URL 编码后拼接。
    """
    et = int(time.time()) + TOKEN_TTL
    method = "sha1"
    res = f"products/{product_id}"
    sign_str = f"{et}\n{method}\n{res}\n{AUTH_VERSION}"
    key_bytes = base64.b64decode(access_key)  # 官方要求：accessKey 先 base64decode
    sign = base64.b64encode(
        hmac.new(key_bytes, sign_str.encode("utf-8"), hashlib.sha1).digest()
    ).decode("ascii")
    # res 中的 "/" 必须编码为 %2F（urllib.parse.quote 默认 safe='/'，需显式清空）
    return (
        f"version={AUTH_VERSION}&res={urllib.parse.quote(res, safe='')}"
        f"&et={et}&method={method}&sign={urllib.parse.quote(sign, safe='')}"
    )


def fetch_latest(access_key, timeout=HTTP_TIMEOUT):
    """从 OneNET 拉取设备最新属性，返回 {temperature, voltage, humidity}。

    每次请求用产品级 accessKey 现算安全鉴权 token；云端返回缺字段或数值非法时抛 ValueError；
    网络失败抛 urllib.error.URLError。
    """
    url = f"{ONENET_API}?product_id={PRODUCT_ID}&device_name={DEVICE_NAME}"
    token = build_auth_token(access_key, PRODUCT_ID)
    req = urllib.request.Request(url, headers={"Authorization": token})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    # 业务码非 0（如鉴权失败、设备不存在）时，把 Token 与 URL 一并带出，便于逐字核对
    if payload.get("code") not in (0, None, "0"):
        raise ValueError(
            f"云端返回错误码 {payload.get('code')}：{payload.get('msg') or payload}\n"
            f"请求 URL: {url}\n"
            f"生成的 Token: {token}"
        )

    # 兼容不同响应包装：官方返回平铺属性数组 [{identifier, value}] / 含 properties 的对象 / 直接属性字典
    data = payload.get("data")
    props = {}
    if isinstance(data, dict):
        lst = data.get("list") or data.get("properties")
        if isinstance(lst, list):
            for p in lst:
                props[p.get("identifier")] = p.get("value")
        else:
            props = {
                k: v for k, v in data.items()
                if k not in ("product_id", "device_name")
            }
    elif isinstance(data, list):
        for item in data or []:
            if isinstance(item, dict) and "identifier" in item:
                props[item.get("identifier")] = item.get("value")
            else:
                for p in (item or {}).get("properties") or []:
                    props[p.get("identifier")] = p.get("value")

    values = {}
    missing = []
    for key in PROPERTY_KEYS:
        v = props.get(key)
        if v is None:
            missing.append(key)
            continue
        try:
            values[key] = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"属性 {key} 不是有效数值：{v!r}")
    if missing:
        raise ValueError(f"云端最新数据缺少属性：{'、'.join(missing)}")
    return values


def push_webhook(webhook, values, timeout=HTTP_TIMEOUT):
    """把三个属性 POST 到本地 /onenet/webhook，返回响应 JSON。失败抛异常。"""
    req = urllib.request.Request(
        webhook,
        data=json.dumps(values).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def insert_local(values):
    """webhook 不可达时的兜底：直接写入 sensor.db（source_type=ONENET），返回自增 id。"""
    import db  # 延迟导入，正常路径仍走 webhook 统一入库
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return db.insert_sample(
        timestamp, values["temperature"], values["humidity"], values["voltage"], "ONENET"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=3.0, help="云端拉取间隔（秒），默认 3")
    parser.add_argument("--webhook", default=DEFAULT_WEBHOOK, help=f"本地 webhook 地址，默认 {DEFAULT_WEBHOOK}")
    args = parser.parse_args()

    access_key = (os.getenv("ONENET_ACCESS_KEY") or "").strip()
    if not access_key:
        logger.error("未配置 ONENET_ACCESS_KEY（.env 中应存在，来源于 OneNET 控制台「安全鉴权」页），拒绝启动。")
        sys.exit(1)
    try:
        base64.b64decode(access_key)  # 控制台生成的 accessKey 应为 base64 编码的密钥
    except Exception:
        logger.error("ONENET_ACCESS_KEY 不是合法的 base64 编码，请核对是否复制完整（控制台「安全鉴权」页的完整 accessKey）。")
        sys.exit(1)
    if not PRODUCT_ID or not DEVICE_NAME:
        logger.error("未配置 ONENET_PRODUCT_ID / ONENET_DEVICE_NAME（.env 中应存在），拒绝启动。")
        sys.exit(1)
    if args.interval < 1:
        logger.error("拉取间隔必须 >= 1 秒")
        sys.exit(1)

    logger.info(
        "启动（产品级安全鉴权）：产品 %s / 设备 %s，每 %.1f 秒拉取一次并推送到 %s",
        PRODUCT_ID, DEVICE_NAME, args.interval, args.webhook,
    )
    round_no = 0
    while True:
        round_no += 1
        cycle_start = time.time()
        try:
            values = fetch_latest(access_key)
            logger.info(
                "第 %d 轮：云端拉取成功，温度 %.1f °C / 电压 %.2f V / 湿度 %.1f %%",
                round_no, values["temperature"], values["voltage"], values["humidity"],
            )
            try:
                resp = push_webhook(args.webhook, values)
                logger.info(
                    "第 %d 轮：已推送到本地 webhook（返回 %s），本轮耗时 %.2f 秒",
                    round_no,
                    resp.get("status") if isinstance(resp, dict) else resp,
                    time.time() - cycle_start,
                )
            except Exception as exc:
                # 本地 web 服务未启动等场景：绕过 webhook 直接写 sensor.db，保证数据不丢
                row_id = insert_local(values)
                logger.warning(
                    "第 %d 轮：webhook 不可达（%r），已直接写入 sensor.db（id=%s）",
                    round_no, exc, row_id,
                )
        except urllib.error.HTTPError as exc:
            logger.warning("第 %d 轮：云端接口返回 HTTP %s：%s", round_no, exc.code, exc.read()[:200])
        except urllib.error.URLError as exc:
            logger.warning("第 %d 轮：网络请求失败（云端）：%s", round_no, exc.reason)
        except ValueError as exc:
            logger.warning("第 %d 轮：%s", round_no, exc)
        except Exception as exc:
            logger.warning("第 %d 轮：未预期异常：%r", round_no, exc)
        # 扣除本轮耗时后再睡，保证相邻两次拉取间隔稳定在 3 秒附近
        time.sleep(max(0.0, args.interval - (time.time() - cycle_start)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("已手动停止。")
