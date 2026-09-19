"""
信度分析（Cronbach's α 系数）
前端稳定输出版
"""

import pandas as pd
import numpy as np


def cronbach_alpha(df):
    k = df.shape[1]
    if k < 2:
        return float('nan')
    item_var = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float('nan')
    return (k / (k - 1)) * (1 - item_var.sum() / total_var)


def cronbach_alpha_standardized(df):
    k = df.shape[1]
    if k < 2:
        return float('nan')
    corr_matrix = df.corr()
    r_bar = corr_matrix.values[np.triu_indices(k, k=1)].mean()
    return (k * r_bar) / (1 + (k - 1) * r_bar)


def reliability_analysis(df, items=None):
    if items:
        df = df[items].copy()
    df = df.dropna()
    n, k = len(df), df.shape[1]

    summary_table = pd.DataFrame({
        "Cronbach's α系数": [cronbach_alpha(df)],
        "标准化Cronbach's α系数": [cronbach_alpha_standardized(df)],
        "项数": [k],
        "样本数": [n],
    })

    records = []
    for col in df.columns:
        rest = df.drop(columns=[col])
        rest_sum = rest.sum(axis=1)
        records.append({
            "题项": col,
            "删除项后的平均值": rest_sum.mean(),
            "删除项后的方差": rest_sum.var(ddof=1),
            "删除的项与删除项后的总体的相关性": df[col].corr(rest_sum),
            "删除项后的Cronbach's α系数": cronbach_alpha(rest),
        })
    item_deleted_table = pd.DataFrame(records)

    meta = {
        'method': 'Reliability Analysis',
        'analysis_sample_size': int(n),
        'item_count': int(k),
        'items': df.columns.tolist(),
        'table_keys': ['summary_table', 'item_deleted_table']
    }

    display = {
        'tables': {
            'summary_table': {
                'title': "Cronbach's α系数表",
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['单行摘要表，直接横向展示即可。']
            },
            'item_deleted_table': {
                'title': '删除分析题统计汇总表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通明细表，无需合并单元格。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': {
            'summary_table': summary_table,
            'item_deleted_table': item_deleted_table,
        },
        'display': display,
    }


if __name__ == '__main__':
    DATA_PATH = 'your_data.xlsx'
    ITEMS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    df = pd.read_excel(DATA_PATH)
    result = reliability_analysis(df, items=ITEMS)
    print(result['tables']['summary_table'].to_string(index=False))
    print(result['tables']['item_deleted_table'].to_string(index=False))
