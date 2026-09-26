"""
线性回归分析工具
前端稳定输出版，不生成图片
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ._numeric import NA, numeric_display, significance_star


warnings.filterwarnings('ignore')


def _star(p):
    """显著性星号统一由 _numeric.significance_star 定义，见那里的口径说明。"""

    return significance_star(p)


def _pstr(p):
    try:
        pf = float(p)
    except Exception:
        return '—'
    s = '<0.001' if pf < 0.001 else f'{pf:.3f}'
    return s + _star(pf)


def linear_regression(data_or_path, dependent, independents, save_path=None):
    all_cols = [dependent] + independents
    if isinstance(data_or_path, pd.DataFrame):
        data = data_or_path[all_cols].dropna().reset_index(drop=True)
    else:
        raw = pd.read_csv(data_or_path) if str(data_or_path).endswith('.csv') else pd.read_excel(data_or_path)
        data = raw[all_cols].dropna().reset_index(drop=True)

    n = len(data)
    Y = data[dependent].values
    X_raw = data[independents].values
    X = sm.add_constant(X_raw)

    model = sm.OLS(Y, X).fit()
    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    f_val = model.fvalue
    f_p = model.f_pvalue

    std_y = np.std(Y, ddof=1)
    std_x = np.std(X_raw, axis=0, ddof=1)
    betas = params[1:] * std_x / std_y
    vifs = [variance_inflation_factor(X, i + 1) for i in range(X_raw.shape[1])]

    # ---- 构建回归系数表（不含 R²/调整R²/F 列，这些放到合并的模型汇总行中）----
    # 常数项没有 Beta 和 VIF，这里存 NaN 而不是 '-'：一个占位字符串会把整列拖
    # 成文本，连带让这一列其余的真数值也变成字符串。渲染成 '-' 是展示层的事。
    coef_cols = ['因变量', '自变量', 'B', '标准误', 'Beta', 't', 'P值', '显著性', 'VIF']

    rows = [{
        '因变量': dependent,
        '自变量': '常数',
        'B': params[0],
        '标准误': bse[0],
        'Beta': NA,
        't': tvals[0],
        'P值': float(pvals[0]),
        '显著性': _star(float(pvals[0])),
        'VIF': NA,
    }]

    for i, var in enumerate(independents):
        rows.append({
            '因变量': dependent if i == 0 else '',
            '自变量': var,
            'B': params[i + 1],
            '标准误': bse[i + 1],
            'Beta': betas[i],
            't': tvals[i + 1],
            'P值': float(pvals[i + 1]),
            '显著性': _star(float(pvals[i + 1])),
            'VIF': vifs[i],
        })

    regression_table = pd.DataFrame(rows, columns=coef_cols)

    # ---- 模型级统计量 ----
    # 这些数曾经只存在于下面那句展示文本里：R²、F 值、自由度、p 全都是渲染后的
    # 字符串，结构化结果中没有。报告模块要用就只能解析文本或自己重算，而重算
    # 是项目约定明令禁止的。数值是事实，文本是它的渲染。
    df_model = len(independents)
    df_resid = n - len(independents) - 1  # 残差自由度
    model_summary_table = pd.DataFrame(
        [
            {
                '因变量': dependent,
                'R²': float(r2),
                '调整R²': float(adj_r2),
                'F': float(f_val),
                '模型自由度': int(df_model),
                '残差自由度': int(df_resid),
                'P值': float(f_p),
                '样本数': int(n),
            }
        ],
        columns=['因变量', 'R²', '调整R²', 'F', '模型自由度', '残差自由度', 'P值', '样本数'],
    )

    model_summary_text = (
        f"模型汇总：R² = {r2:.3f}，"
        f"调整R² = {adj_r2:.3f}，"
        f"F({df_model}, {df_resid}) = {f_val:.3f}，"
        f"p = {_pstr(f_p)}"
    )

    meta = {
        'method': 'Linear Regression',
        'analysis_sample_size': int(n),
        'dependent': dependent,
        'independents': independents,
        'table_keys': ['model_summary_table', 'regression_table']
    }

    display = {
        'tables': {
            'model_summary_table': {
                'title': '模型汇总',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['结构化模型级统计量，供报告和二次计算使用。']
            },
            'regression_table': {
                'title': '线性回归分析结果',
                'model_summary_row': model_summary_text,
                'merge_cells': [
                    {'field': '因变量', 'merge_consecutive_same_values': True, 'align': 'center', 'vertical_align': 'middle'},
                ],
                'orientation': 'wide',
                **numeric_display(na_text='-', combine={'P值': '显著性'}),
                'notes': ['常数项的 Beta 与 VIF 为 NaN，按 numeric.na_text 渲染成 "-"。',
                          'P值 为完整精度数值，显著性星号在“显著性”列，渲染时拼接即可。',
                          'model_summary_row 由 model_summary_table 派生，仅供展示，不要从中解析数值。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': {
            'model_summary_table': model_summary_table,
            'regression_table': regression_table,
        },
        'display': display,
    }


run_regression = linear_regression


if __name__ == '__main__':
    result = linear_regression(
        data_or_path='your_data.xlsx',
        dependent='Q1',
        independents=['Q2', 'Q3'],
    )
    print(result['tables']['regression_table'].to_string(index=False))
