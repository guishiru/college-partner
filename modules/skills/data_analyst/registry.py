"""Language metadata for the Data Analyst skill.

This registry describes how user language maps to a canonical method. It does
not contain statistical calculations; those remain behind the analysis engine.
"""

METHOD_SKILLS = {
    "descriptive": {
        "label": "描述统计",
        "keywords": ["描述统计", "描述性统计", "均值", "标准差", "中位数"],
        "kind": "cols",
        "min_cols": 1,
        "numeric": True,
        "question": "请指定要做描述统计的变量。",
    },
    "frequency": {
        "label": "频数分析",
        "keywords": ["频数", "频率", "占比", "百分比", "分布"],
        "kind": "cols",
        "min_cols": 1,
        "numeric": False,
        "question": "请指定要查看频数或分布的变量。",
    },
    "reliability": {
        "label": "信度分析",
        "keywords": ["信度", "可靠性", "Cronbach", "Alpha", "α"],
        "kind": "cols",
        "min_cols": 2,
        "numeric": True,
        "question": "信度分析需要同一量表或维度下至少 2 个题项，请指定题项。",
    },
    "validity": {
        "label": "效度分析",
        "keywords": ["效度", "KMO", "Bartlett", "巴特利特"],
        "kind": "cols",
        "min_cols": 2,
        "numeric": True,
        "question": "效度分析需要至少 2 个数值题项，请指定题项。",
    },
    "correlation": {
        "label": "相关分析",
        "keywords": ["相关分析", "相关性", "Pearson", "皮尔逊"],
        "kind": "cols",
        "min_cols": 2,
        "numeric": True,
        "question": "相关分析至少需要 2 个数值变量，请指定变量。",
    },
    "efa": {
        "label": "探索性因子分析",
        "keywords": ["探索性因子", "EFA", "因子载荷", "提取因子"],
        "kind": "cols",
        "min_cols": 2,
        "numeric": True,
        "question": "EFA 至少需要 2 个数值题项，请指定题项。",
    },
    "cfa": {
        "label": "验证性因子分析",
        "keywords": ["验证性因子", "CFA", "测量模型"],
        "kind": "factors",
        "question": "CFA 需要因子结构，请按“维度名：题项1、题项2”提供。",
    },
    "regression": {
        "label": "线性回归",
        "keywords": ["线性回归", "回归", "因变量", "自变量"],
        "kind": "regression",
        "question": "线性回归需要明确因变量 Y 和至少 1 个自变量 X。",
    },
    "moderation": {
        "label": "调节效应分析",
        "keywords": ["调节效应", "调节", "交互效应", "交互项"],
        "kind": "moderation",
        "question": "调节效应需要明确自变量 X、因变量 Y 和调节变量 M。",
    },
    "parallel_mediation": {
        "label": "平行中介效应",
        "keywords": ["平行中介", "中介效应", "中介分析", "Bootstrap中介"],
        "kind": "parallel_mediation",
        "question": "中介效应需要明确自变量 X、因变量 Y 和至少 1 个中介变量。",
    },
    "sem": {
        "label": "结构方程模型",
        "keywords": ["结构方程", "SEM", "路径模型", "路径分析"],
        "kind": "sem",
        "question": "SEM 需要明确因子结构和至少 1 条路径关系。",
    },
}


def detect_methods(text: str) -> list[str]:
    lowered = (text or "").lower()
    matches = []
    for method, spec in METHOD_SKILLS.items():
        if any(keyword.lower() in lowered for keyword in spec["keywords"]):
            matches.append(method)
    return matches
