"""
效度分析（KMO + Bartlett 球形度检验）
前端稳定输出版
"""

import pandas as pd
import numpy as np
from scipy import stats


def _p_with_stars(p):
    if p < 0.001:
        return '<0.001***'
    if p < 0.01:
        return f'{p:.3f}***'
    if p < 0.05:
        return f'{p:.3f}**'
    if p < 0.1:
        return f'{p:.3f}*'
    return f'{p:.3f}'


def kmo_test(df):
    corr = df.corr().values
    corr_inv = np.linalg.inv(corr)
    n = corr.shape[0]
    partial_corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                partial_corr[i, j] = -corr_inv[i, j] / np.sqrt(corr_inv[i, i] * corr_inv[j, j])
    r_sq = sum(corr[i, j] ** 2 for i in range(n) for j in range(n) if i != j)
    p_sq = sum(partial_corr[i, j] ** 2 for i in range(n) for j in range(n) if i != j)
    return r_sq / (r_sq + p_sq)


def bartlett_test(df):
    n, k = df.shape
    corr = df.corr().values
    chi2 = -(n - 1 - (2 * k + 5) / 6) * np.log(np.linalg.det(corr))
    df_val = int(k * (k - 1) / 2)
    p_val = 1 - stats.chi2.cdf(chi2, df_val)
    return chi2, df_val, p_val


def validity_analysis(df, items=None):
    if items:
        df = df[items].copy()
    df = df.dropna()
    kmo = kmo_test(df)
    chi2, df_val, p_val = bartlett_test(df)

    validity_table = pd.DataFrame([
        {'检验项': 'KMO检验', '子项': 'KMO值', '值': round(kmo, 3)},
        {'检验项': 'Bartlett球形度检验', '子项': '近似卡方', '值': round(chi2, 3)},
        {'检验项': 'Bartlett球形度检验', '子项': 'df', '值': df_val},
        {'检验项': 'Bartlett球形度检验', '子项': 'P', '值': _p_with_stars(p_val)},
    ])

    meta = {
        'method': 'Validity Analysis',
        'analysis_sample_size': int(len(df)),
        'item_count': int(df.shape[1]),
        'items': df.columns.tolist(),
        'table_keys': ['validity_table']
    }

    display = {
        'tables': {
            'validity_table': {
                'title': 'KMO检验和Bartlett的检验',
                'merge_cells': [
                    {'field': '检验项', 'merge_consecutive_same_values': True, 'align': 'center', 'vertical_align': 'middle'}
                ],
                'orientation': 'grouped',
                'notes': ['按“检验项”列纵向合并相邻相同单元格。', '“Bartlett球形度检验”三行合并为一个单元格并居中展示。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': {'validity_table': validity_table},
        'display': display,
    }


if __name__ == '__main__':
    DATA_PATH = 'your_data.xlsx'
    ITEMS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    df = pd.read_excel(DATA_PATH)
    result = validity_analysis(df, items=ITEMS)
    print(result['tables']['validity_table'].to_string(index=False))
