from vector_store import search_vector_store


# =========================================================
# 1. Retrieval Evaluation Test Set
# =========================================================
#
# expected_sources：
# 一个问题可能允许多个合理来源。
#
# 例如：
#
# “会哪些 AI 技术？”
#
# skills.md 是直接答案，
# projects.md 里也包含 AI 技术实践，
# 因此两个来源都可以视为合理命中。
# =========================================================

TEST_CASES = [

    # -----------------------------------------------------
    # 教育背景
    # -----------------------------------------------------

    {
        "query":
            "高颂岩的硕士学校和专业是什么？",

        "expected_sources":
            ["education.md"],
    },

    {
        "query":
            "高颂岩本科是什么专业？",

        "expected_sources":
            ["education.md"],
    },


    # -----------------------------------------------------
    # 科研经历
    # -----------------------------------------------------

    {
        "query":
            "高颂岩的硕士课题是什么？",

        "expected_sources":
            ["research.md"],
    },

    {
        "query":
            "高颂岩的鱼鳞研究主要做了什么？",

        "expected_sources":
            ["research.md"],
    },

    {
        "query":
            "高颂岩发表过什么论文？",

        "expected_sources":
            ["research.md"],
    },


    # -----------------------------------------------------
    # 实习经历
    # -----------------------------------------------------

    {
        "query":
            "高颂岩有哪些实习经历？",

        "expected_sources":
            ["internships.md"],
    },

    {
        "query":
            "高颂岩在九颂仁华实习时做了什么？",

        "expected_sources":
            ["internships.md"],
    },

    {
        "query":
            "高颂岩在301医院主要做了什么？",

        "expected_sources":
            ["internships.md"],
    },


    # -----------------------------------------------------
    # 项目经历
    # -----------------------------------------------------

    {
        "query":
            "高颂岩做过哪些AI项目？",

        "expected_sources":
            ["projects.md"],
    },

    {
        "query":
            "高颂岩的AI Resume Agent是怎么实现的？",

        "expected_sources":
            ["projects.md"],
    },

    {
        "query":
            "科研文献智能分析Agent主要实现什么功能？",

        "expected_sources":
            ["projects.md"],
    },


    # -----------------------------------------------------
    # 技能与能力
    # -----------------------------------------------------

    {
        "query":
            "高颂岩会哪些AI开发技术？",

        "expected_sources":
            [
                "skills.md",
                "projects.md",
            ],
    },

    {
        "query":
            "高颂岩会哪些材料表征技术？",

        "expected_sources":
            [
                "skills.md",
                "research.md",
            ],
    },

    {
        "query":
            "高颂岩会哪些数据分析和科研绘图工具？",

        "expected_sources":
            ["skills.md"],
    },


    # -----------------------------------------------------
    # 跨类别问题
    #
    # 这种问题不应该被简单 Metadata Filter 锁死。
    # -----------------------------------------------------

    {
        "query":
            "高颂岩有哪些项目管理和团队协作经验？",

        "expected_sources":
            [
                "skills.md",
                "projects.md",
            ],
    },
]


# =========================================================
# 2. 判断一次 Retrieval 是否命中
# =========================================================

def is_hit(
    results,
    expected_sources,
    k
):
    """
    判断 Top-K 中是否出现至少一个正确来源。
    """

    top_k_results = results[:k]

    retrieved_sources = [

        result["source"]

        for result
        in top_k_results
    ]


    return any(

        source
        in expected_sources

        for source
        in retrieved_sources
    )


# =========================================================
# 3. 获取 Top-K Sources
# =========================================================

def get_top_sources(
    results,
    k=5
):

    return [

        result["source"]

        for result
        in results[:k]
    ]


# =========================================================
# 4. 单种 Retrieval Strategy Evaluation
# =========================================================

def evaluate_strategy(
    strategy_name,
    use_intent_filter
):
    """
    对一种 Retrieval Strategy
    计算：

    Hit@1
    Hit@3
    Hit@5
    """

    total = len(
        TEST_CASES
    )

    hit_1 = 0
    hit_3 = 0
    hit_5 = 0

    bad_cases = []


    print()
    print("=" * 70)

    print(
        f"Evaluation Strategy："
        f"{strategy_name}"
    )

    print("=" * 70)


    for index, test_case in enumerate(
        TEST_CASES,
        start=1
    ):

        query = (
            test_case[
                "query"
            ]
        )

        expected_sources = (
            test_case[
                "expected_sources"
            ]
        )


        # =================================================
        # Retrieval
        # =================================================

        results = search_vector_store(

            query=query,

            # 为了同时计算 Hit@1 / @3 / @5
            top_k=5,

            use_intent_filter=
                use_intent_filter,

            debug=False,
        )


        # =================================================
        # Metrics
        # =================================================

        current_hit_1 = (
            is_hit(
                results,
                expected_sources,
                1
            )
        )

        current_hit_3 = (
            is_hit(
                results,
                expected_sources,
                3
            )
        )

        current_hit_5 = (
            is_hit(
                results,
                expected_sources,
                5
            )
        )


        if current_hit_1:
            hit_1 += 1

        if current_hit_3:
            hit_3 += 1

        if current_hit_5:
            hit_5 += 1


        # =================================================
        # 输出每一个 Test Case
        # =================================================

        top_sources = (
            get_top_sources(
                results,
                5
            )
        )


        print()
        print(
            f"[{index:02d}] "
            f"{query}"
        )

        print(
            "Expected：",
            expected_sources
        )

        print(
            "Top-5：",
            top_sources
        )

        print(
            "Hit@1：",
            "✅"
            if current_hit_1
            else "❌"
        )

        print(
            "Hit@3：",
            "✅"
            if current_hit_3
            else "❌"
        )

        print(
            "Hit@5：",
            "✅"
            if current_hit_5
            else "❌"
        )


        # =================================================
        # 保存 Top-1 Bad Case
        # =================================================

        if not current_hit_1:

            bad_cases.append(
                {
                    "query":
                        query,

                    "expected":
                        expected_sources,

                    "top_sources":
                        top_sources,

                    "top_sections":
                        [
                            result[
                                "section_title"
                            ]
                            for result
                            in results[:5]
                        ],
                }
            )


    # =====================================================
    # 最终分数
    # =====================================================

    hit_1_rate = (
        hit_1 / total
    )

    hit_3_rate = (
        hit_3 / total
    )

    hit_5_rate = (
        hit_5 / total
    )


    print()
    print("=" * 70)

    print(
        f"{strategy_name} Results"
    )

    print("=" * 70)

    print(
        f"Hit@1："
        f"{hit_1}/{total} "
        f"= {hit_1_rate:.1%}"
    )

    print(
        f"Hit@3："
        f"{hit_3}/{total} "
        f"= {hit_3_rate:.1%}"
    )

    print(
        f"Hit@5："
        f"{hit_5}/{total} "
        f"= {hit_5_rate:.1%}"
    )


    # =====================================================
    # Bad Cases
    # =====================================================

    print()
    print(
        "Top-1 Bad Cases："
    )


    if not bad_cases:

        print(
            "无 Top-1 Bad Case。"
        )


    else:

        for case in bad_cases:

            print()
            print("-" * 70)

            print(
                "Query：",
                case[
                    "query"
                ]
            )

            print(
                "Expected：",
                case[
                    "expected"
                ]
            )

            print(
                "Top Sources：",
                case[
                    "top_sources"
                ]
            )

            print(
                "Top Sections：",
                case[
                    "top_sections"
                ]
            )


    # =====================================================
    # 返回结果，供最终比较
    # =====================================================

    return {
        "name":
            strategy_name,

        "hit_1":
            hit_1,

        "hit_3":
            hit_3,

        "hit_5":
            hit_5,

        "total":
            total,

        "hit_1_rate":
            hit_1_rate,

        "hit_3_rate":
            hit_3_rate,

        "hit_5_rate":
            hit_5_rate,

        "bad_cases":
            bad_cases,
    }


# =========================================================
# 5. 对比两种 Retrieval Strategy
# =========================================================

def compare_results(
    pure_result,
    intent_result
):

    print()
    print()
    print("=" * 70)
    print("Retrieval Strategy Comparison")
    print("=" * 70)

    print()

    print(
        "Pure Vector Search"
    )

    print(
        f"Hit@1："
        f"{pure_result['hit_1_rate']:.1%}"
    )

    print(
        f"Hit@3："
        f"{pure_result['hit_3_rate']:.1%}"
    )

    print(
        f"Hit@5："
        f"{pure_result['hit_5_rate']:.1%}"
    )


    print()

    print(
        "Intent-Aware Retrieval"
    )

    print(
        f"Hit@1："
        f"{intent_result['hit_1_rate']:.1%}"
    )

    print(
        f"Hit@3："
        f"{intent_result['hit_3_rate']:.1%}"
    )

    print(
        f"Hit@5："
        f"{intent_result['hit_5_rate']:.1%}"
    )


    # =====================================================
    # Improvement
    # =====================================================

    hit_1_improvement = (
        intent_result[
            "hit_1_rate"
        ]
        -
        pure_result[
            "hit_1_rate"
        ]
    )

    hit_3_improvement = (
        intent_result[
            "hit_3_rate"
        ]
        -
        pure_result[
            "hit_3_rate"
        ]
    )

    hit_5_improvement = (
        intent_result[
            "hit_5_rate"
        ]
        -
        pure_result[
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
# 6. Main
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("AI Resume RAG Retrieval Evaluation")
    print("=" * 70)

    print(
        f"Test Cases："
        f"{len(TEST_CASES)}"
    )


    # =====================================================
    # A. 原始 Pure Vector Search
    # =====================================================

    pure_result = evaluate_strategy(

        strategy_name=
            "Pure Vector Search",

        use_intent_filter=
            False,
    )


    # =====================================================
    # B. Intent-Aware Retrieval
    # =====================================================

    intent_result = evaluate_strategy(

        strategy_name=
            "Intent-Aware Retrieval",

        use_intent_filter=
            True,
    )


    # =====================================================
    # C. 最终对比
    # =====================================================

    compare_results(
        pure_result,
        intent_result
    )