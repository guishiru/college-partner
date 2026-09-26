"""
Exploratory Factor Analysis (EFA)
PCA extraction + Varimax rotation
"""

import numpy as np
import pandas as pd

from ._numeric import NA, numeric_display, significance_star



def _varimax(loadings, max_iter=1000, tol=1e-8, kaiser_normalize=True):
    """Varimax rotation with optional Kaiser normalization (SPSS-style)."""
    A = loadings.copy().astype(float)
    p, k = A.shape
    if k < 2:
        return A

    if kaiser_normalize:
        row_norm = np.sqrt(np.sum(A ** 2, axis=1))
        row_norm = np.where(row_norm == 0, 1.0, row_norm)
        A = A / row_norm[:, None]
    else:
        row_norm = np.ones(p)

    R = np.eye(k)
    d_old = 0.0
    for _ in range(max_iter):
        Lambda = A @ R
        u, s, vh = np.linalg.svd(
            A.T @ (Lambda ** 3 - (Lambda @ np.diag(np.diag(Lambda.T @ Lambda))) / p)
        )
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and d - d_old < tol:
            break
        d_old = d

    rotated = A @ R
    if kaiser_normalize:
        rotated = rotated * row_norm[:, None]
    return rotated


def _format_p_value(p):
    if p is None or pd.isna(p):
        return None
    p = float(p)
    if p < 0.001:
        return "<0.001"
    return round(p, 4)


def _calculate_kmo_and_bartlett(df):
    """Calculate KMO and Bartlett's test if factor_analyzer is available."""
    try:
        from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

        chi_square_value, bartlett_p = calculate_bartlett_sphericity(df)
        kmo_all, kmo_model = calculate_kmo(df)
        return {
            'kmo': round(float(kmo_model), 4),
            'bartlett_chi_square': round(float(chi_square_value), 4),
            'bartlett_p_value': _format_p_value(bartlett_p)
        }
    except Exception:
        return {
            'kmo': None,
            'bartlett_chi_square': None,
            'bartlett_p_value': None
        }


def exploratory_factor_analysis(df, items=None, n_factors=None):
    """
    EFA with PCA extraction and Varimax rotation

    Parameters:
    -----------
    df : pd.DataFrame
    items : list of column names (optional)
    n_factors : int (optional, if None will auto-determine)

    Returns:
    --------
    dict with stable meta + tables structure for frontend/report rendering.
    """
    if items:
        df = df[items].copy()

    original_sample_size = len(df)

    # Handle missing values
    df = df.dropna()
    analysis_sample_size = len(df)

    # Standardize
    from scipy import stats
    data = stats.zscore(df.values)

    # Correlation matrix
    corr = np.corrcoef(data.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Determine number of factors
    auto_factor_count = int(sum(eigenvalues > 1))
    if n_factors is None:
        n_factors = auto_factor_count
    n_factors = int(n_factors)

    # PCA extraction
    unrotated_loadings = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])

    # Varimax rotation
    rotated = _varimax(unrotated_loadings)

    # Variance explained
    variance_pct_all = eigenvalues / sum(eigenvalues) * 100
    cumulative_pct_all = np.cumsum(variance_pct_all)

    # Create DataFrames
    factor_names = [f'F{i+1}' for i in range(n_factors)]
    component_matrix_table = pd.DataFrame(unrotated_loadings, index=df.columns, columns=factor_names)
    rotated_component_matrix_table = pd.DataFrame(rotated, index=df.columns, columns=factor_names)

    communalities_initial = np.ones(len(df.columns))
    communalities_extraction = np.sum(unrotated_loadings ** 2, axis=1)
    communalities_rotation = np.sum(rotated ** 2, axis=1)
    communality_table = pd.DataFrame({
        '变量': df.columns,
        '初始': np.round(communalities_initial, 4),
        '提取': np.round(communalities_extraction, 4),
        '旋转后': np.round(communalities_rotation, 4)
    })

    eigenvalue_rows = []
    for i, eigenvalue in enumerate(eigenvalues, start=1):
        eigenvalue_rows.append({
            '成分': f'F{i}',
            '特征根': round(float(eigenvalue), 4),
            '方差解释率(%)': round(float(variance_pct_all[i - 1]), 4),
            '累计方差解释率(%)': round(float(cumulative_pct_all[i - 1]), 4),
            '是否保留': '是' if i <= n_factors else '否'
        })
    eigenvalue_table = pd.DataFrame(eigenvalue_rows)

    total_variance_rows = []
    retained_eigen_sum = float(np.sum(eigenvalues[:n_factors]))
    retained_rotated_variance = np.sum(rotated ** 2, axis=0) / len(df.columns) * 100
    retained_unrotated_variance = variance_pct_all[:n_factors]
    retained_rotated_cumulative = np.cumsum(retained_rotated_variance)
    retained_unrotated_cumulative = np.cumsum(retained_unrotated_variance)

    for i in range(n_factors):
        total_variance_rows.append({
            '成分': f'F{i+1}',
            '提取特征根': round(float(eigenvalues[i]), 4),
            '提取方差解释率(%)': round(float(retained_unrotated_variance[i]), 4),
            '提取累计方差解释率(%)': round(float(retained_unrotated_cumulative[i]), 4),
            '旋转后平方载荷和': round(float(np.sum(rotated[:, i] ** 2)), 4),
            '旋转后方差解释率(%)': round(float(retained_rotated_variance[i]), 4),
            '旋转后累计方差解释率(%)': round(float(retained_rotated_cumulative[i]), 4)
        })
    total_variance_table = pd.DataFrame(total_variance_rows)

    total_variance_display_rows = []
    for i, eigenvalue in enumerate(eigenvalues, start=1):
        row = {
            '成分': i,
            '特征根': round(float(eigenvalue), 3),
            '方差解释率(%)': round(float(variance_pct_all[i - 1]), 3),
            '累积方差解释率(%)': round(float(cumulative_pct_all[i - 1]), 3),
            # 未保留的成分没有旋转后结果。存 NaN 而不是空串：空串会把整列
            # 拖成文本，连保留成分的真数值也一起变成字符串。
            '旋转后特征根': NA,
            '旋转后方差解释率(%)': NA,
            '旋转后累积方差解释率(%)': NA,
        }
        if i <= n_factors:
            row['旋转后特征根'] = round(float(np.sum(rotated[:, i - 1] ** 2)), 3)
            row['旋转后方差解释率(%)'] = round(float(retained_rotated_variance[i - 1]), 3)
            row['旋转后累积方差解释率(%)'] = round(float(retained_rotated_cumulative[i - 1]), 3)
        total_variance_display_rows.append(row)
    total_variance_display_table = pd.DataFrame(total_variance_display_rows)

    rotated_component_display_table_data = {'名称': df.columns}
    for idx in range(n_factors):
        rotated_component_display_table_data[f'旋转后因子载荷系数{idx + 1}'] = np.round(rotated[:, idx], 3)
    rotated_component_display_table_data['共同度（公因子方差）'] = np.round(communalities_rotation, 3)
    rotated_component_display_table = pd.DataFrame(rotated_component_display_table_data)

    # 成分得分系数矩阵：R⁻¹ × 旋转后载荷（SPSS Component Score Coefficient Matrix 口径）
    score_coef = np.linalg.inv(corr) @ rotated
    component_score_display_table_data = {'名称': df.columns}
    for idx in range(n_factors):
        component_score_display_table_data[f'成分{idx + 1}'] = np.round(score_coef[:, idx], 3)
    component_score_display_table = pd.DataFrame(component_score_display_table_data)

    factor_summary_display_rows = []
    total_retained_variance = float(np.sum(retained_rotated_variance)) if n_factors > 0 else 0.0
    cumulative_rotated_variance = np.cumsum(retained_rotated_variance)
    for idx in range(n_factors):
        factor_summary_display_rows.append({
            '名称': f'因子{idx + 1}',
            '旋转后方差解释率(%)': round(float(retained_rotated_variance[idx]), 3),
            '旋转后累积方差解释率(%)': round(float(cumulative_rotated_variance[idx]), 3),
            '权重(%)': round(float(retained_rotated_variance[idx] / total_retained_variance * 100), 3) if total_retained_variance else None
        })
    factor_summary_display_table = pd.DataFrame(factor_summary_display_rows)

    communality_display_table = rotated_component_display_table[['名称', '共同度（公因子方差）']].copy()

    factor_assignment = rotated_component_matrix_table.abs().idxmax(axis=1)
    factor_max_loading = rotated_component_matrix_table.abs().max(axis=1)
    factor_summary_rows = []
    for factor in factor_names:
        assigned_items = factor_assignment[factor_assignment == factor].index.tolist()
        factor_summary_rows.append({
            '因子': factor,
            '题项数量': len(assigned_items),
            '题项列表': '、'.join(assigned_items),
            '平均绝对载荷': round(float(factor_max_loading[factor_assignment == factor].mean()), 4) if assigned_items else None,
            '方差解释率(%)': round(float(retained_rotated_variance[factor_names.index(factor)]), 4)
        })
    factor_summary_rows.append({
        '因子': '汇总',
        '题项数量': int(len(df.columns)),
        '题项列表': '、'.join(df.columns.tolist()),
        '平均绝对载荷': round(float(factor_max_loading.mean()), 4),
        '方差解释率(%)': round(float(np.sum(retained_rotated_variance)), 4)
    })
    factor_summary_rows.append({
        '因子': '分析样本量',
        '题项数量': int(analysis_sample_size),
        '题项列表': '',
        '平均绝对载荷': None,
        '方差解释率(%)': None
    })
    factor_summary_table = pd.DataFrame(factor_summary_rows)

    standardized_df = pd.DataFrame(data, index=df.index, columns=df.columns)
    factor_score_values = standardized_df.values @ score_coef
    factor_score_table = pd.DataFrame(
        np.round(factor_score_values, 4),
        index=df.index,
        columns=[f'{name}_score' for name in factor_names]
    )
    factor_score_table.index.name = '样本'
    factor_score_table = factor_score_table.reset_index()

    rotated_variance_weight = retained_rotated_variance / np.sum(retained_rotated_variance)
    composite_score_values = factor_score_values @ rotated_variance_weight.reshape(-1, 1)
    composite_score_table = pd.DataFrame({
        '样本': df.index,
        '综合得分': np.round(composite_score_values.flatten(), 4)
    })
    composite_score_display_table = composite_score_table.copy()
    composite_score_display_table['样本'] = np.arange(1, len(composite_score_display_table) + 1)
    composite_score_display_table = composite_score_display_table.rename(columns={'样本': '排序'})
    composite_score_display_table['行索引'] = df.reset_index(drop=True).index + 1
    composite_score_display_table['综合得分'] = composite_score_display_table['综合得分'].round(3)
    original_value_table = df.reset_index(drop=True).copy()
    composite_score_display_table = pd.concat([composite_score_display_table[['排序', '行索引', '综合得分']], original_value_table], axis=1)
    composite_score_display_table = composite_score_display_table.sort_values('综合得分', ascending=False).reset_index(drop=True)
    composite_score_display_table['排序'] = np.arange(1, len(composite_score_display_table) + 1)

    diagnostics = _calculate_kmo_and_bartlett(df)

    meta = {
        'method': 'EFA (PCA + Varimax)',
        'items': df.columns.tolist(),
        'item_count': int(len(df.columns)),
        'original_sample_size': int(original_sample_size),
        'analysis_sample_size': int(analysis_sample_size),
        'missing_rows_dropped': int(original_sample_size - analysis_sample_size),
        'factor_count': int(n_factors),
        'auto_factor_count_by_eigen_gt_1': int(auto_factor_count),
        'total_variance_explained_pct': round(float(np.sum(retained_rotated_variance)), 4),
        'kmo': diagnostics['kmo'],
        'bartlett_chi_square': diagnostics['bartlett_chi_square'],
        'bartlett_p_value': diagnostics['bartlett_p_value']
    }

    tables = {
        'eigenvalue_table': eigenvalue_table,
        'total_variance_table': total_variance_table,
        'total_variance_display_table': total_variance_display_table,
        'component_matrix_table': component_matrix_table.round(4).reset_index().rename(columns={'index': '变量'}),
        'rotated_component_matrix_table': rotated_component_matrix_table.round(4).reset_index().rename(columns={'index': '变量'}),
        'rotated_component_display_table': rotated_component_display_table,
        'component_score_display_table': component_score_display_table,
        'communality_table': communality_table,
        'communality_display_table': communality_display_table,
        'factor_summary_table': factor_summary_table,
        'factor_summary_display_table': factor_summary_display_table,
        'factor_score_table': factor_score_table,
        'composite_score_table': composite_score_table,
        'composite_score_display_table': composite_score_display_table
    }

    display = {
        'tables': {
            'eigenvalue_table': {
                'title': '特征根表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，按成分逐行展示即可。']
            },
            'total_variance_table': {
                'title': '总方差解释表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，展示提取前后及旋转后的方差解释结果。']
            },
            'total_variance_display_table': {
                'title': '总方差解释表（展示版）',
                **numeric_display(na_text='-'),
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['按截图样式整理，未保留成分的旋转后列留空。']
            },
            'component_matrix_table': {
                'title': '成分矩阵',
                'merge_cells': [],
                'orientation': 'matrix',
                'notes': ['首列“变量”为题项名，其余列为各因子载荷。']
            },
            'rotated_component_matrix_table': {
                'title': '旋转后的成分矩阵',
                'merge_cells': [],
                'orientation': 'matrix',
                'notes': ['首列“变量”为题项名，其余列为旋转载荷。']
            },
            'rotated_component_display_table': {
                'title': '旋转后因子载荷系数与共同度表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['按截图样式展示旋转后因子载荷系数与共同度。']
            },
            'component_score_display_table': {
                'title': '成分矩阵表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['按截图样式展示成分1载荷。']
            },
            'communality_table': {
                'title': '共同度表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，无需合并单元格。']
            },
            'communality_display_table': {
                'title': '旋转后因子载荷系数表-共同度',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['按截图样式单独展示共同度列。']
            },
            'factor_summary_table': {
                'title': '因子汇总表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['按因子逐行展示题项归属与方差解释率。']
            },
            'factor_summary_display_table': {
                'title': '因子提取分析',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['按截图样式展示各因子的方差解释率与权重。']
            },
            'factor_score_table': {
                'title': '因子得分表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['该表通常供程序消费，如前端展示建议分页。']
            },
            'composite_score_table': {
                'title': '综合得分表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['该表通常供程序消费或导出。']
            },
            'composite_score_display_table': {
                'title': '综合得分表（展示版）',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['按截图样式同时展示综合得分与原始题项值。']
            }
        }
    }

    meta['table_keys'] = list(tables.keys())

    return {
        'meta': meta,
        'tables': tables,
        'display': display,
    }


def print_efa_results(result):
    """CLI/debug print only, not for business logic."""
    print("【EFA 元信息】")
    print(pd.DataFrame([result['meta']]).to_string(index=False))

    print("\n【特征根表】")
    print(result['tables']['eigenvalue_table'].to_string(index=False))

    print("\n【总方差解释表】")
    print(result['tables']['total_variance_table'].to_string(index=False))

    print("\n【旋转成分矩阵】")
    print(result['tables']['rotated_component_matrix_table'].to_string(index=False))

    print("\n【共同度表】")
    print(result['tables']['communality_table'].to_string(index=False))

    print("\n【因子汇总表】")
    print(result['tables']['factor_summary_table'].to_string(index=False))


if __name__ == "__main__":
    import sys
    df = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else 'data.xlsx')
    items = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    n_factors = int(sys.argv[3]) if len(sys.argv) > 3 else None
    result = exploratory_factor_analysis(df, items, n_factors)
    print_efa_results(result)
