"""
Correlation Analysis
前端稳定输出版
"""

import pandas as pd
from scipy import stats


def _star(p):
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.1:
        return '*'
    return ''


def _p_display(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def correlation_analysis(df, items=None, method='pearson'):
    if items is not None:
        df = df[items].copy()
    df = df.dropna()
    cols = df.columns.tolist()

    matrix = pd.DataFrame(index=cols, columns=cols, dtype=object)

    for col_i in cols:
        for col_j in cols:
            if col_i == col_j:
                matrix.loc[col_i, col_j] = '1.000'
            else:
                if method == 'pearson':
                    r, p = stats.pearsonr(df[col_i], df[col_j])
                elif method == 'spearman':
                    r, p = stats.spearmanr(df[col_i], df[col_j])
                elif method == 'kendall':
                    r, p = stats.kendalltau(df[col_i], df[col_j])
                else:
                    raise ValueError(f'Unsupported method: {method}')

                display = f'{r:.3f}({_p_display(p)}{_star(p)})'
                matrix.loc[col_i, col_j] = display

    correlation_display_matrix = matrix.reset_index().rename(columns={'index': ''})

    meta = {
        'method': 'Correlation Analysis',
        'analysis_sample_size': int(len(df)),
        'item_count': int(df.shape[1]),
        'items': cols,
        'correlation_method': method,
        'table_keys': ['correlation_display_matrix']
    }

    display = {
        'tables': {
            'correlation_display_matrix': {
                'title': '相关性矩阵',
                'merge_cells': [],
                'orientation': 'matrix',
                'notes': ['渲染为矩阵表，首列为空表头，用于显示行变量名。', '单元格直接显示 display 字符串即可。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': {
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
