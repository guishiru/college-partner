"""
Descriptive Statistics Analysis
前端稳定输出版
"""

import pandas as pd
import numpy as np


def descriptive_analysis(df, items=None):
    if items:
        df = df[items].copy()
    df = df.dropna()

    descriptive_table = pd.DataFrame({
        '变量名': df.columns,
        '样本量': [len(df)] * len(df.columns),
        '最大值': df.max().values,
        '最小值': df.min().values,
        '平均值': df.mean().values,
        '标准差': df.std().values,
        '中位数': df.median().values,
        '方差': df.var().values,
        '偏度': df.skew().values,
        '峰度': df.kurtosis().values,
        '变异系数（CV）': (df.std() / df.mean()).replace([np.inf, -np.inf], np.nan).values,
    })

    meta = {
        'method': 'Descriptive Analysis',
        'analysis_sample_size': int(len(df)),
        'item_count': int(df.shape[1]),
        'items': df.columns.tolist(),
        'table_keys': ['descriptive_table']
    }

    display = {
        'tables': {
            'descriptive_table': {
                'title': '描述性分析',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，首列为变量名，无需合并单元格。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': {'descriptive_table': descriptive_table},
        'display': display,
    }


if __name__ == '__main__':
    import sys
    df = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else 'data.xlsx')
    items = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    result = descriptive_analysis(df, items)
    print(result['tables']['descriptive_table'].to_string(index=False))
