from vector_store import search_vector_store
from evaluation.reranker import rerank_results


# =========================================================
# 1. Section-Level Test Set
# =========================================================

TEST_CASES = [

    # -----------------------------------------------------
    # 教育背景
    # -----------------------------------------------------

    {
        "query":
            "高颂岩的硕士学校和专业是什么？",

        "expected_source":
            "education.md",

        "expected_section_keywords":
            ["硕士教育"],
    },

    {
        "query":
            "高颂岩本科是什么专业？",

        "expected_source":
            "education.md",

        "expected_section_keywords":
            ["本科教育"],
    },


    # -----------------------------------------------------
    # 科研经历
    # -----------------------------------------------------

    {
        "query":
            "高颂岩的硕士课题是什么？",

        "expected_source":
            "research.md",

        "expected_section_keywords":
            ["硕士课题"],
    },

    {
        "query":
            "高颂岩的鱼鳞研究主要做了什么？",

        "expected_source":
            "research.md",

        "expected_section_keywords":
            ["天然鱼鳞"],
    },

    {
        "query":
            "高颂岩发表的 Composites Part B 论文是什么？",

        "expected_source":
            "research.md",

        "expected_section_keywords":
            [
                "天然鱼鳞",
                "科研成果",
            ],
    },


    # -----------------------------------------------------
    # 实习经历
    # -----------------------------------------------------

    {
        "query":
            "高颂岩在九颂仁华实习时做了什么？",

        "expected_source":
            "internships.md",

        "expected_section_keywords":
            ["九颂仁华"],
    },

    {
        "query":
            "高颂岩在301医院实习时主要做了什么？",

        "expected_source":
            "internships.md",

        "expected_section_keywords":
            ["301 医院"],
    },


    # -----------------------------------------------------
    # 项目经历
    # -----------------------------------------------------

    {
        "query":
            "高颂岩的 AI Resume Agent 是怎么实现的？",

        "expected_source":
            "projects.md",

        "expected_section_keywords":
            ["AI Resume Agent"],
    },

    {
        "query":
            "科研文献智能分析 Agent 主要实现什么功能？",

        "expected_source":
            "projects.md",

        "expected_section_keywords":
            ["科研文献智能分析 Agent"],
    },

    {
        "query":
            "碧海琼硅项目主要做了什么？",

        "expected_source":
            "projects.md",

        "expected_section_keywords":
            ["碧海琼硅"],
    },

    {
        "query":
            "3D智造快速制模项目主要做了什么？",

        "expected_source":
            "projects.md",

        "expected_section_keywords":
            ["3D 智造"],
    },
]


# =========================================================
# 2. 判断单条 Result 是否正确
# =========================================================

def result_is_correct(
    result,
    expected_source,
    expected_section_keywords,
):
    """
    必须同时满足：

    1. source 正确
    2. section_title 命中任意一个预期关键词

    才算 Section-Level 正确。
    """

    if result["source"] != expected_source:
        return False

    section_title = (
        result["section_title"]
        .lower()
        .replace(" ", "")
    )

    for keyword in expected_section_keywords:

        normalized_keyword = (
            keyword
            .lower()
            .replace(" ", "")
        )

        if normalized_keyword in section_title:
            return True

    return False


# =========================================================
# 3. Hit@K
# =========================================================

def is_hit_at_k(
    results,
    expected_source,
    expected_section_keywords,
    k,
):
    """
    Top-K 中只要存在一个正确 Section，
    就视为 Hit。
    """

    for result in results[:k]:

        if result_is_correct(
            result,
            expected_source,
            expected_section_keywords,
        ):
            return True

    return False


# =========================================================
# 4. 打印一个 Ranking
# =========================================================

def print_ranking(
    results,
    expected_source,
    expected_section_keywords,
    use_rerank_score=False,
):
    """
    打印当前 Top-K 排名。

    Before:
        显示 Chroma distance

    After:
        显示 Jina rerank_score
    """

    for rank, result in enumerate(
        results,
        start=1,
    ):

        correct = result_is_correct(
            result,
            expected_source,
            expected_section_keywords,
        )

        marker = (
            "✅"
            if correct
            else "  "
        )


        if use_rerank_score:

            score_text = (
                f"rerank_score="
                f"{result['rerank_score']:.6f}"
            )

        else:

            score_text = (
                f"distance="
                f"{result['distance']:.4f}"
            )


        print(
            f"{rank}. "
            f"{marker} "
            f"{result['source']} | "
            f"{result['section_title']} | "
            f"{score_text}"
        )


# =========================================================
# 5. 初始化 Metrics
# =========================================================

def create_metrics():

    return {
        "hit_1": 0,
        "hit_3": 0,
        "hit_5": 0,
        "bad_cases": [],
    }


# =========================================================
# 6. 更新 Metrics
# =========================================================

def update_metrics(
    metrics,
    results,
    test_case,
):

    expected_source = (
        test_case[
            "expected_source"
        ]
    )

    expected_sections = (
        test_case[
            "expected_section_keywords"
        ]
    )


    hit_1 = is_hit_at_k(
        results,
        expected_source,
        expected_sections,
        1,
    )

    hit_3 = is_hit_at_k(
        results,
        expected_source,
        expected_sections,
        3,
    )

    hit_5 = is_hit_at_k(
        results,
        expected_source,
        expected_sections,
        5,
    )


    if hit_1:
        metrics["hit_1"] += 1

    if hit_3:
        metrics["hit_3"] += 1

    if hit_5:
        metrics["hit_5"] += 1


    if not hit_1:

        metrics[
            "bad_cases"
        ].append(
            {
                "query":
                    test_case["query"],

                "expected_source":
                    expected_source,

                "expected_sections":
                    expected_sections,

                "top_results":
                    [
                        {
                            "source":
                                result["source"],

                            "section":
                                result[
                                    "section_title"
                                ],
                        }

                        for result
                        in results
                    ],
            }
        )


    return (
        hit_1,
        hit_3,
        hit_5,
    )


# =========================================================
# 7. 打印 Metrics
# =========================================================

def print_metrics(
    title,
    metrics,
    total,
):

    print()
    print("=" * 70)

    print(
        title
    )

    print("=" * 70)


    hit_1_rate = (
        metrics["hit_1"]
        / total
    )

    hit_3_rate = (
        metrics["hit_3"]
        / total
    )

    hit_5_rate = (
        metrics["hit_5"]
        / total
    )


    print(
        f"Section Hit@1："
        f"{metrics['hit_1']}/{total} "
        f"= {hit_1_rate:.1%}"
    )

    print(
        f"Section Hit@3："
        f"{metrics['hit_3']}/{total} "
        f"= {hit_3_rate:.1%}"
    )

    print(
        f"Section Hit@5："
        f"{metrics['hit_5']}/{total} "
        f"= {hit_5_rate:.1%}"
    )


    print()
    print(
        "Top-1 Bad Cases："
    )


    if not metrics["bad_cases"]:

        print(
            "无 Top-1 Bad Case。"
        )

    else:

        for case in metrics[
            "bad_cases"
        ]:

            print()
            print("-" * 70)

            print(
                "Query：",
                case["query"]
            )

            print(
                "Expected Source：",
                case[
                    "expected_source"
                ]
            )

            print(
                "Expected Section：",
                case[
                    "expected_sections"
                ]
            )

            print(
                "Top Results："
            )


            for rank, result in enumerate(
                case[
                    "top_results"
                ],
                start=1,
            ):

                print(
                    f"{rank}. "
                    f"{result['source']} | "
                    f"{result['section']}"
                )


    return {
        "hit_1_rate":
            hit_1_rate,

        "hit_3_rate":
            hit_3_rate,

        "hit_5_rate":
            hit_5_rate,
    }


# =========================================================
# 8. Main Evaluation
# =========================================================

def evaluate():

    total = len(
        TEST_CASES
    )


    before_metrics = (
        create_metrics()
    )

    after_metrics = (
        create_metrics()
    )


    print()
    print("=" * 70)
    print(
        "Section-Level Retrieval + Jina Reranker Evaluation"
    )
    print("=" * 70)

    print(
        f"Test Cases：{total}"
    )


    # =====================================================
    # 逐个 Test Case
    # =====================================================

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        query = (
            test_case["query"]
        )

        expected_source = (
            test_case[
                "expected_source"
            ]
        )

        expected_sections = (
            test_case[
                "expected_section_keywords"
            ]
        )


        print()
        print()
        print("#" * 70)

        print(
            f"[{index:02d}] {query}"
        )

        print("#" * 70)

        print(
            "Expected Source：",
            expected_source
        )

        print(
            "Expected Section：",
            expected_sections
        )


        # =================================================
        # A. Chroma Recall
        # =================================================

        recall_results = (
            search_vector_store(

                query=query,

                top_k=5,

                use_intent_filter=True,

                debug=False,
            )
        )


        print()
        print(
            "Before Jina Reranker"
        )

        print("-" * 70)


        print_ranking(

            recall_results,

            expected_source,

            expected_sections,

            use_rerank_score=False,
        )


        (
            before_hit_1,
            before_hit_3,
            before_hit_5,
        ) = update_metrics(

            before_metrics,

            recall_results,

            test_case,
        )


        print()

        print(
            "Before Hit@1：",
            "✅"
            if before_hit_1
            else "❌"
        )

        print(
            "Before Hit@3：",
            "✅"
            if before_hit_3
            else "❌"
        )

        print(
            "Before Hit@5：",
            "✅"
            if before_hit_5
            else "❌"
        )


        # =================================================
        # B. Jina Reranker
        # =================================================

        reranked_results = (
            rerank_results(

                query=query,

                results=
                    recall_results,

                top_k=5,
            )
        )


        print()
        print(
            "After Jina Reranker"
        )

        print("-" * 70)


        print_ranking(

            reranked_results,

            expected_source,

            expected_sections,

            use_rerank_score=True,
        )


        (
            after_hit_1,
            after_hit_3,
            after_hit_5,
        ) = update_metrics(

            after_metrics,

            reranked_results,

            test_case,
        )


        print()

        print(
            "After Hit@1：",
            "✅"
            if after_hit_1
            else "❌"
        )

        print(
            "After Hit@3：",
            "✅"
            if after_hit_3
            else "❌"
        )

        print(
            "After Hit@5：",
            "✅"
            if after_hit_5
            else "❌"
        )


    # =====================================================
    # 9. 汇总 Before
    # =====================================================

    before_result = (
        print_metrics(

            title=
                "Before Jina Reranker Results",

            metrics=
                before_metrics,

            total=
                total,
        )
    )


    # =====================================================
    # 10. 汇总 After
    # =====================================================

    after_result = (
        print_metrics(

            title=
                "After Jina Reranker Results",

            metrics=
                after_metrics,

            total=
                total,
        )
    )


    # =====================================================
    # 11. 最终 Comparison
    # =====================================================

    print()
    print()
    print("=" * 70)

    print(
        "Section-Level Reranker Comparison"
    )

    print("=" * 70)


    print()
    print(
        "Before Jina Reranker"
    )

    print(
        f"Hit@1："
        f"{before_result['hit_1_rate']:.1%}"
    )

    print(
        f"Hit@3："
        f"{before_result['hit_3_rate']:.1%}"
    )

    print(
        f"Hit@5："
        f"{before_result['hit_5_rate']:.1%}"
    )


    print()
    print(
        "After Jina Reranker"
    )

    print(
        f"Hit@1："
        f"{after_result['hit_1_rate']:.1%}"
    )

    print(
        f"Hit@3："
        f"{after_result['hit_3_rate']:.1%}"
    )

    print(
        f"Hit@5："
        f"{after_result['hit_5_rate']:.1%}"
    )


    # =====================================================
    # Improvement
    # =====================================================

    hit_1_improvement = (
        after_result[
            "hit_1_rate"
        ]
        -
        before_result[
            "hit_1_rate"
        ]
    )

    hit_3_improvement = (
        after_result[
            "hit_3_rate"
        ]
        -
        before_result[
            "hit_3_rate"
        ]
    )

    hit_5_improvement = (
        after_result[
            "hit_5_rate"
        ]
        -
        before_result[
            "hit_5_rate"
        ]
    )


    print()
    print(
        "Improvement"
    )

    print(
        f"Hit@1："
        f"{hit_1_improvement:+.1%}"
    )

    print(
        f"Hit@3："
        f"{hit_3_improvement:+.1%}"
    )

    print(
        f"Hit@5："
        f"{hit_5_improvement:+.1%}"
    )


# =========================================================
# 12. Main
# =========================================================

if __name__ == "__main__":

    evaluate()