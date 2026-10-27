"""轻量级 RAG：txt 知识库 -> 段落切分 -> TF-IDF 检索 -> 拼接 Prompt。

不用向量数据库：中文按字符二元组做词项，手写 TF-IDF + 余弦相似度，
语料规模小（几十个 chunk）时足够准确。
"""
import logging
import math
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger("rag")

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

# 余弦相似度下限：低于此值的 chunk 视为不相关。
# 实测：真实命中（如"热失控前兆"）远高于纯噪声，取 0.05 居中。
MIN_SCORE = 0.05

_chunks = None


def load_knowledge():
    """读取 knowledge/*.txt，按段落（空行分隔）切分成 chunks。"""
    global _chunks
    if _chunks is not None:
        return _chunks
    chunks = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for i, para in enumerate(paragraphs, 1):
            # 无句末标点的段落视为标题行，不参与检索（短文本会在 TF-IDF 中虚高）
            if not re.search(r"[。！？]", para):
                continue
            chunks.append({"source": path.name, "paragraph": i, "text": para})
    logger.info("Loaded %d chunks from %s", len(chunks), KNOWLEDGE_DIR)
    _chunks = chunks
    return _chunks


def _bigrams(text):
    """中文分词成本高，用字符二元组近似词项（如“电池”-> 电池/池过）。"""
    clean = re.sub(r"\s+", "", text)
    return [clean[i:i + 2] for i in range(len(clean) - 1)]


def retrieve(query, k=2):
    """TF-IDF + 余弦相似度，返回最相关的 k 个 chunk：[(score, chunk), ...]，按分数降序。"""
    chunks = load_knowledge()
    if not chunks:
        return []

    # 文档频率：词项出现在多少个 chunk 中
    df = Counter()
    for chunk in chunks:
        for term in set(_bigrams(chunk["text"])):
            df[term] += 1
    n = len(chunks)

    def tfidf_vector(text):
        tf = Counter(_bigrams(text))
        vec = {}
        for term, count in tf.items():
            idf = math.log((n + 1) / (df[term] + 1)) + 1
            vec[term] = (1 + math.log(count)) * idf
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    qvec = tfidf_vector(query)
    scored = []
    for chunk in chunks:
        cvec = tfidf_vector(chunk["text"])
        score = sum(w * cvec.get(t, 0.0) for t, w in qvec.items())
        scored.append((score, chunk))
    scored.sort(key=lambda item: -item[0])
    top = [(score, chunk) for score, chunk in scored[:k] if score >= MIN_SCORE]
    logger.info("Retrieved %d chunks for query: %s", len(top), query[:40])
    return top


def build_rag_prompt(question, top_chunks):
    """把检索到的 chunk 拼进 Prompt，要求基于资料回答并注明来源。"""
    if not top_chunks:
        return (
            "你是锂电池安全知识助手。\n"
            f"用户问题：{question}\n\n"
            "本次检索没有找到任何相关资料。资料不足只能回答资料不足，不得编造，"
            "也不得给出任何建议或补充信息。请只回答：资料不足。"
        )
    refs = []
    for i, (_, chunk) in enumerate(top_chunks, 1):
        refs.append(f"[{i}] 来源：{chunk['source']}（第 {chunk['paragraph']} 段）\n{chunk['text']}")
    return (
        "你是锂电池安全知识助手。请必须根据提供的参考资料回答用户问题，"
        "并在回答中注明引用来源编号（如 [1]）。\n"
        "规则：\n"
        "1. 只能使用参考资料中的信息，不得使用参考资料之外的知识。\n"
        "2. 资料不足只能回答资料不足，不得编造：如果参考资料没有直接涉及用户问题的主题"
        "（例如用户问电池起火、但资料未提及起火），或信息不足以回答问题，"
        "必须只回答“资料不足”，不得推测、扩展或使用自身知识补充。\n\n"
        "参考资料：\n" + "\n\n".join(refs) + f"\n\n用户问题：{question}"
    )




