from vector_store import (
    ensure_vector_store,
    search_vector_store
)


# =========================================================
# RAG 检索测试集
#
# expected_sources:
# 一个问题可能有多个合理知识来源
# =========================================================

TEST_CASES = [
    {
        "question": "高颂岩做过哪些 AI 项目？",
        "expected_sources": [
            "projects.md",
            "skills.md"
        ]
    },
    {
        "question": "高颂岩的 AI 简历知识库助手是怎么实现的？",
        "expected_sources": [
            "projects.md"
        ]
    },
    {
        "question": "高颂岩会哪些 AI 和软件开发技术？",
        "expected_sources": [
            "skills.md"
        ]
    },
    {
        "question": "高颂岩会哪些数据分析和科研制图工具？",
        "expected_sources": [
            "skills.md"
        ]
    },
    {
        "question": "高颂岩的求职方向是什么？",
        "expected_sources": [
            "education.md",
            "skills.md"
        ]
    },
    {
        "question": "高颂岩是什么专业背景？",
        "expected_sources": [
            "education.md"
        ]
    },
    {
        "question": "高颂岩的硕士课题是什么？",
        "expected_sources": [
            "research.md"
        ]
    },
    {
        "question": "高颂岩做过哪些科研工作？",
        "expected_sources": [
            "research.md",
            "projects.md"
        ]
    },
    {
        "question": "高颂岩有没有材料表征经验？",
        "expected_sources": [
            "skills.md",
            "research.md"
        ]
    },
    {
        "question": "高颂岩有哪些项目管理或团队协作经验？",
        "expected_sources": [
            "projects.md",
            "skills.md"
        ]
    }
]


# =========================================================
# 判断 Hit@K
# =========================================================

def is_hit(
    results,
    expected_sources,
    k
):

    top_k_results = results[:k]

    retrieved_sources = [
        result["source"]
        for result in top_k_results
    ]

    return any(
        source in retrieved_sources
        for source in expected_sources
    )


# =========================================================
# 运行评测
# =========================================================

def evaluate():

    ensure_vector_store()

    total = len(TEST_CASES)

    hit_1 = 0
    hit_3 = 0
    hit_5 = 0

    bad_cases = []


    print("\n")
    print("=" * 60)
    print("RAG Retrieval Evaluation")
    print("=" * 60)


    for index, test_case in enumerate(
        TEST_CASES,
        start=1
    ):

        question = test_case["question"]

        expected_sources = test_case[
            "expected_sources"
        ]


        # =================================================
        # 检索 Top-5
        # =================================================

        results = search_vector_store(
            query=question,
            top_k=5
        )


        # =================================================
        # Hit@K
        # =================================================

        hit1 = is_hit(
            results,
            expected_sources,
            1
        )

        hit3 = is_hit(
            results,
            expected_sources,
            3
        )

        hit5 = is_hit(
            results,
            expected_sources,
            5
        )


        if hit1:
            hit_1 += 1

        if hit3:
            hit_3 += 1

        if hit5:
            hit_5 += 1


        # =================================================
        # 输出当前测试
        # =================================================

        print("\n")
        print("-" * 60)

        print(
            f"测试 {index}/{total}"
        )

        print(
            f"问题：{question}"
        )

        print(
            "合理来源：",
            ", ".join(expected_sources)
        )


        print("\n实际 Top-5：")


        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{rank}. "
                f"{result['source']} "
                f"(distance="
                f"{result['distance']:.4f})"
            )


        print(
            f"\nHit@1："
            f"{'✅' if hit1 else '❌'}"
        )

        print(
            f"Hit@3："
            f"{'✅' if hit3 else '❌'}"
        )

        print(
            f"Hit@5："
            f"{'✅' if hit5 else '❌'}"
        )


        # =================================================
        # Bad Case
        # =================================================

        if not hit1:

            bad_cases.append(
                {
                    "question": question,
                    "expected": expected_sources,
                    "actual": (
                        results[0]["source"]
                        if results
                        else "无结果"
                    )
                }
            )


    # =====================================================
    # 最终结果
    # =====================================================

    hit_1_rate = hit_1 / total
    hit_3_rate = hit_3 / total
    hit_5_rate = hit_5 / total


    print("\n")
    print("=" * 60)
    print("最终评测结果")
    print("=" * 60)


    print(
        f"测试问题数量：{total}"
    )

    print(
        f"Hit@1："
        f"{hit_1}/{total} "
        f"= {hit_1_rate:.2%}"
    )

    print(
        f"Hit@3："
        f"{hit_3}/{total} "
        f"= {hit_3_rate:.2%}"
    )

    print(
        f"Hit@5："
        f"{hit_5}/{total} "
        f"= {hit_5_rate:.2%}"
    )


    # =====================================================
    # Bad Case 分析
    # =====================================================

    print("\n")
    print("=" * 60)
    print("Bad Cases（Top-1 未命中）")
    print("=" * 60)


    if not bad_cases:

        print(
            "没有 Bad Case，"
            "所有问题 Top-1 均命中。"
        )

    else:

        for index, case in enumerate(
            bad_cases,
            start=1
        ):

            print(
                f"\nBad Case {index}"
            )

            print(
                "问题：",
                case["question"]
            )

            print(
                "合理来源：",
                ", ".join(
                    case["expected"]
                )
            )

            print(
                "Top-1 实际来源：",
                case["actual"]
            )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    evaluate()