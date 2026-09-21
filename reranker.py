from functools import lru_cache


# =========================================================
# 1. Reranker 配置
# =========================================================

RERANKER_MODEL_NAME = (
    "jinaai/jina-reranker-v2-base-multilingual"
)


# =========================================================
# 2. Lazy Loading
# =========================================================

@lru_cache(maxsize=1)
def get_reranker_model():
    """
    懒加载 Jina Reranker。

    只有第一次真正调用 rerank_results()
    时才加载模型。

    后续调用直接复用同一个模型，
    不会重复加载。
    """

    print(
        f"首次需要 Reranker，"
        f"正在加载：{RERANKER_MODEL_NAME}"
    )

    # 放在函数内部导入，
    # 避免 import reranker 时就加载
    # sentence-transformers 相关模块。
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        RERANKER_MODEL_NAME,
        trust_remote_code=True,
    )

    print(
        "Jina Reranker 加载完成。"
    )

    return model


# =========================================================
# 3. Rerank
# =========================================================

def rerank_results(
    query: str,
    results: list,
    top_k=None,
):
    """
    对 Chroma 已召回的候选 Chunk 进行重新排序。

    注意：
    调用这个函数时，
    才会真正加载 Jina 模型。

    Pipeline：

    Query
        ↓
    Chroma Recall
        ↓
    Jina Cross-Encoder
        ↓
    Final Ranking
    """

    if not results:
        return []


    # =====================================================
    # 第一次调用才加载模型
    # =====================================================

    reranker_model = (
        get_reranker_model()
    )


    # =====================================================
    # Query + Chunk Pair
    #
    # 使用 Contextual Chunk：
    #
    # 候选人
    # 资料类型
    # 章节
    # 原始正文
    # =====================================================

    pairs = [

        (
            query,
            result["content"]
        )

        for result in results
    ]


    # =====================================================
    # Jina 打分
    # =====================================================

    scores = (
        reranker_model.predict(
            pairs
        )
    )


    # =====================================================
    # 合并结果
    # =====================================================

    reranked_results = []


    for result, score in zip(
        results,
        scores,
    ):

        new_result = (
            result.copy()
        )

        new_result[
            "rerank_score"
        ] = float(score)

        reranked_results.append(
            new_result
        )


    # =====================================================
    # Jina 分数越高越相关
    # =====================================================

    reranked_results.sort(
        key=lambda item:
            item["rerank_score"],
        reverse=True,
    )


    # =====================================================
    # 最终 Top-K
    # =====================================================

    if top_k is not None:

        return (
            reranked_results[
                :top_k
            ]
        )


    return reranked_results


# =========================================================
# 4. 本地测试
# =========================================================

if __name__ == "__main__":

    from vector_store import (
        search_vector_store
    )


    print()
    print("=" * 70)
    print("Jina Lazy Loading Test")
    print("=" * 70)


    query = input(
        "\n请输入测试问题："
    )


    # =====================================================
    # Chroma Recall
    # =====================================================

    recall_results = (
        search_vector_store(
            query=query,
            top_k=5,
            use_intent_filter=True,
            debug=True,
        )
    )


    print()
    print("=" * 70)
    print("Before Reranker")
    print("=" * 70)


    for rank, result in enumerate(
        recall_results,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{result['source']} | "
            f"{result['section_title']} | "
            f"distance="
            f"{result['distance']:.4f}"
        )


    # =====================================================
    # 这里才第一次加载 Jina
    # =====================================================

    reranked_results = (
        rerank_results(
            query=query,
            results=recall_results,
            top_k=5,
        )
    )


    print()
    print("=" * 70)
    print("After Jina Reranker")
    print("=" * 70)


    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{result['source']} | "
            f"{result['section_title']} | "
            f"rerank_score="
            f"{result['rerank_score']:.6f}"
        )