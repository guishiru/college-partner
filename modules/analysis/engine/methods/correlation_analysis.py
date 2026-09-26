"""
Correlation Analysis
前端稳定输出版
"""

import pandas as pd

from scipy import stats

from ._numeric import significance_star


def _star(p):
    """显著性星号统一由 _numeric.significance_star 定义，见那里的口径说明。"""

    return significance_star(p)


def _p_display(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def correlation_analysis(df, items=None, method='pearson'):
    if items is not None:
        df = df[items].copy()
    df = df.dropna()
    cols = df.columns.tolist()

    # 先算出原始系数和 p 值，展示矩阵是它们的派生物。
    # 反过来做的话，结果里就只剩 '0.566(<0.001***)' 这种字符串：精度丢到三位，
    # 而 '<0.001' 根本不是一个值而是一个界，报告模块无论怎么解析都拿不回真值。
    coefficients: dict[tuple[str, str], float] = {}
    p_values: dict[tuple[str, str], float] = {}

    for index_i, col_i in enumerate(cols):
        for col_j in cols[index_i + 1:]:
            if method == 'pearson':
                r, p = stats.pearsonr(df[col_i], df[col_j])
            elif method == 'spearman':
                r, p = stats.spearmanr(df[col_i], df[col_j])
            elif method == 'kendall':
                r, p = stats.kendalltau(df[col_i], df[col_j])
            else:
                raise ValueError(f'Unsupported method: {method}')
            coefficients[(col_i, col_j)] = float(r)
            p_values[(col_i, col_j)] = float(p)

    def _pair(col_i, col_j):
        return (col_i, col_j) if (col_i, col_j) in coefficients else (col_j, col_i)

    # 结构化结果：每对变量一行，系数和 p 值各自是完整精度的数值列。
    correlation_table = pd.DataFrame(
        [
            {
                '变量1': col_i,
                '变量2': col_j,
                '相关系数': coefficients[(col_i, col_j)],
                'P值': p_values[(col_i, col_j)],
                '显著性标记': _star(p_values[(col_i, col_j)]),
                '样本数': int(len(df)),
            }
            for (col_i, col_j) in coefficients
        ],
        columns=['变量1', '变量2', '相关系数', 'P值', '显著性标记', '样本数'],
    )

    matrix = pd.DataFrame(index=cols, columns=cols, dtype=object)
    for col_i in cols:
        for col_j in cols:
            if col_i == col_j:
                matrix.loc[col_i, col_j] = '1.000'
            else:
                key = _pair(col_i, col_j)
                r, p = coefficients[key], p_values[key]
                matrix.loc[col_i, col_j] = f'{r:.3f}({_p_display(p)}{_star(p)})'

    correlation_display_matrix = matrix.reset_index().rename(columns={'index': ''})

    meta = {
        'method': 'Correlation Analysis',
        'analysis_sample_size': int(len(df)),
        'item_count': int(df.shape[1]),
        'items': cols,
        'correlation_method': method,
        'table_keys': ['correlation_table', 'correlation_display_matrix']
    }

    display = {
        'tables': {
            'correlation_table': {
                'title': '相关分析结果',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['结构化结果表，系数和 P 值为原始数值，供报告和二次计算使用。']
            },
            'correlation_display_matrix': {
                'title': '相关性矩阵',
                'merge_cells': [],
                'orientation': 'matrix',
                'notes': ['渲染为矩阵表，首列为空表头，用于显示行变量名。',
                          '单元格字符串由 correlation_table 派生，仅供展示，不要从中解析数值。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': {
            'correlation_table': correlation_table,
            'correlation_display_matrix': correlation_display_matrix,
        },
        'display': display,
    }


if __name__ == '__main__':
    import sys
    df = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else 'data.xlsx')
    items = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    method = sys.argv[3] if len(sys.argv) > 3 else 'pearson'
    result = correlation_analysis(df, items, method)
    print(result['tables']['correlation_display_matrix'].to_string(index=False))
