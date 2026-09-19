"""
线性回归分析工具
前端稳定输出版，不生成图片
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')


def _star(p):
    try:
        p = float(p)
    except Exception:
        return ''
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.1:
        return '*'
    return ''


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
    coef_cols = ['因变量', '自变量', 'B', '标准误', 'Beta', 't', 'P', 'VIF']

    rows = [{
        '因变量': dependent,
        '自变量': '常数',
        'B': params[0],
        '标准误': bse[0],
        'Beta': '-',
        't': tvals[0],
        'P': _pstr(pvals[0]),
        'VIF': '-',
    }]

    for i, var in enumerate(independents):
        rows.append({
            '因变量': dependent if i == 0 else '',
            '自变量': var,
            'B': params[i + 1],
            '标准误': bse[i + 1],
            'Beta': betas[i],
            't': tvals[i + 1],
            'P': _pstr(pvals[i + 1]),
            'VIF': vifs[i],
        })

    regression_table = pd.DataFrame(rows, columns=coef_cols)

    # ---- 模型汇总信息（用于第一行合并单元格展示）----
    df_resid = n - len(independents) - 1  # 残差自由度
    model_summary_text = (
        f"模型汇总：R² = {r2:.3f}，"
        f"调整R² = {adj_r2:.3f}，"
        f"F({len(independents)}, {df_resid}) = {f_val:.3f}，"
        f"p = {_pstr(f_p)}"
    )

    meta = {
        'method': 'Linear Regression',
        'analysis_sample_size': int(n),
        'dependent': dependent,
        'independents': independents,
        'table_keys': ['regression_table']
    }

    display = {
        'tables': {
            'regression_table': {
                'title': '线性回归分析结果',
                'model_summary_row': model_summary_text,
                'merge_cells': [
                    {'field': '因变量', 'merge_consecutive_same_values': True, 'align': 'center', 'vertical_align': 'middle'},
                ],
                'orientation': 'wide',
                'notes': ['常数项的 Beta 和 VIF 显示为"-"。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': {
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
