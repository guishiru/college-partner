"""
confirmatory_factor_analysis.py
================================
验证性因子分析（CFA） SPSS/Amos 口径
工具：semopy 2.3.11（需 python3.11）
安装依赖：
    python3.11 -m pip install semopy openpyxl pandas numpy scipy

调用示例：
    results = confirmatory_factor_analysis(
        data=df,
        factors={
            'F1': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
            'F2': ['Q11', 'Q12', 'Q13', 'Q14', 'Q15'],
        },
        factor_names={'F1': '因子1', 'F2': '因子2'},   # 可选，因子中文名映射
    )
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# 核心函数
# ──────────────────────────────────────────────

def confirmatory_factor_analysis(data: pd.DataFrame,
                                  factors: dict,
                                  factor_names: dict = None) -> dict:
    """
    执行验证性因子分析（CFA），返回标准化载荷、模型评价、拟合指标等。

    Parameters
    ----------
    data : pd.DataFrame
        观测数据，列名与因子题项对应。
    factors : dict
        因子到题项列名列表的映射，例如：
            {
                'F1': ['Q1', 'Q2', 'Q3'],
                'F2': ['Q4', 'Q5', 'Q6'],
            }
    factor_names : dict, optional
        因子英文名到中文/显示名的映射，例如 {'F1': '认知信任', 'F2': '情感信任'}。
        未提供则使用模型中的原始因子名。

    Returns
    -------
    dict，统一返回 meta + tables + display，并保留旧字段兼容。
    """
    try:
        from semopy import Model
        from semopy.stats import calc_stats
    except ImportError:
        raise ImportError("请先安装 semopy：python3.11 -m pip install semopy")

    analysis_data = data.dropna().copy()

    factor_alias = {factor: f"F{i + 1}" for i, factor in enumerate(factors)}
    all_original_items = []
    for items in factors.values():
        for item in items:
            if item not in all_original_items:
                all_original_items.append(item)
    item_alias = {item: f"V{i + 1}" for i, item in enumerate(all_original_items)}
    alias_to_factor = {alias: factor for factor, alias in factor_alias.items()}
    alias_to_item = {alias: item for item, alias in item_alias.items()}

    missing_items = [item for item in all_original_items if item not in analysis_data.columns]
    if missing_items:
        raise ValueError(f"CFA题项列不存在: {missing_items[:5]}")

    safe_factors = {
        factor_alias[factor]: [item_alias[item] for item in items]
        for factor, items in factors.items()
    }
    safe_data = analysis_data[all_original_items].rename(columns=item_alias)

    model_desc = "\n".join(
        f"{f} =~ {' + '.join(items)}"
        for f, items in safe_factors.items()
    )

    model = Model(model_desc)
    model.fit(safe_data)
    params = model.inspect()

    Sigma_cov = model.calc_sigma()[0]
    sigma_vars = list(model.vars['observed'])
    all_items = sigma_vars
    p = len(all_items)

    factor_list = params[params['op'] == '~']['rval'].unique().tolist()

    factor_sd = {}
    for f in factor_list:
        var_row = params[(params['lval'] == f) & (params['rval'] == f)]
        if len(var_row) > 0:
            factor_sd[f] = np.sqrt(var_row['Estimate'].values[0])
        else:
            factor_sd[f] = 1.0

    load_rows = params[params['op'] == '~'].copy()

    def _std_load(row):
        fsd = factor_sd.get(row['rval'], 1.0)
        idx = all_items.index(row['lval'])
        y_var = Sigma_cov[idx, idx]
        return row['Estimate'] * fsd / np.sqrt(y_var)

    load_rows['std_load'] = load_rows.apply(_std_load, axis=1)

    load_table = load_rows.rename(columns={
        'rval': '因子',
        'lval': '变量',
        'Estimate': '非标准化载荷',
        'std_load': '标准化载荷',
        'Std. Err': '标准误',
        'z-value': 'Z值',
        'p-value': 'P值',
    })[['因子', '变量', '非标准化载荷', '标准化载荷', '标准误', 'Z值', 'P值']].copy()

    load_table['因子'] = load_table['因子'].map(lambda x: factor_names.get(alias_to_factor.get(x, x), alias_to_factor.get(x, x)) if factor_names else alias_to_factor.get(x, x))
    load_table['变量'] = load_table['变量'].map(lambda x: alias_to_item.get(x, x))

    load_table = load_table.reset_index(drop=True)

    eval_rows = []
    for f in factor_list:
        lam = load_rows[load_rows['rval'] == f]['std_load'].values
        lam2 = lam ** 2
        ave = np.mean(lam2)
        sum_lam = np.sum(lam)
        cr = sum_lam ** 2 / (sum_lam ** 2 + np.sum(1 - lam2))
        original_factor = alias_to_factor.get(f, f)
        display_name = factor_names.get(original_factor, original_factor) if factor_names else original_factor
        eval_rows.append({'因子': display_name, 'AVE': ave, 'CR': cr})
    eval_table = pd.DataFrame(eval_rows)

    factor_display = [
        factor_names.get(alias_to_factor.get(f, f), alias_to_factor.get(f, f)) if factor_names else alias_to_factor.get(f, f)
        for f in factor_list
    ]
    n_f = len(factor_list)

    def _sig_star(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ''
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''

    def _fmt_p(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return '-'
        if p < 0.001:
            return '0.000***'
        return f"{p:.3f}{_sig_star(p)}"

    # corr_table 按用户截图口径重算：
    # 对角线展示 √AVE；非对角线展示各因子合成得分之间的 Pearson 相关（附 p 值显著性）。
    factor_score_df = pd.DataFrame(index=analysis_data.index)
    for f in factor_list:
        factor_items = load_rows[load_rows['rval'] == f]['lval'].tolist()
        valid_items = [item for item in factor_items if item in safe_data.columns]
        if valid_items:
            factor_score_df[f] = safe_data[valid_items].mean(axis=1)
        else:
            factor_score_df[f] = np.nan

    corr_table_rows = []
    for i, fi in enumerate(factor_list):
        row_data = {'因子': factor_display[i]}
        for j, fj in enumerate(factor_list):
            display_col = factor_display[j]
            if i == j:
                row_data[display_col] = round(float(np.sqrt(eval_rows[i]['AVE'])), 3)
            else:
                pair_df = factor_score_df[[fi, fj]].dropna()
                if len(pair_df) < 3:
                    row_data[display_col] = '-'
                else:
                    r_val, p_val = stats.pearsonr(pair_df[fi], pair_df[fj])
                    row_data[display_col] = f"{r_val:.3f}({_fmt_p(p_val)})"
        corr_table_rows.append(row_data)

    corr_table = pd.DataFrame(corr_table_rows)

    fit_stats = calc_stats(model)
    fd = fit_stats.iloc[0].to_dict()

    chi2 = fd['chi2']
    df_val = fd['DoF']
    chi2p = fd['chi2 p-value']
    gfi = fd['GFI']
    rmsea = fd['RMSEA']
    cfi = fd['CFI']
    nfi = fd['NFI']
    nnfi = fd['TLI']
    chi2df = chi2 / df_val

    S_obs_cov = np.cov(safe_data[all_items].values.T, ddof=1)
    d_obs = np.sqrt(np.diag(S_obs_cov))
    d_sigma = np.sqrt(np.diag(Sigma_cov))
    S_cor = S_obs_cov / np.outer(d_obs, d_obs)
    Sigma_cor = Sigma_cov / np.outer(d_sigma, d_sigma)
    residual_cor = S_cor - Sigma_cor
    tril_idx = np.tril_indices(p)
    srmr_val = float(np.sqrt(2 * np.sum(residual_cor[tril_idx] ** 2) / (p * (p + 1))))

    residual_cov = S_obs_cov - Sigma_cov
    rmr_val = float(np.sqrt(2 * np.sum(residual_cov[tril_idx] ** 2) / (p * (p + 1))))

    fit_table = pd.DataFrame({
        '指标': ['χ²', 'df', 'P', 'χ²/df', 'GFI', 'RMSEA', 'RMR', 'SRMR', 'CFI', 'NFI', 'NNFI(TLI)'],
        '值': [chi2, int(df_val), chi2p, chi2df, gfi, rmsea, rmr_val, srmr_val, cfi, nfi, nnfi],
        '参考标准': ['—', '—', '>0.05', '<3', '>0.9', '<0.10', '<0.05', '<0.08', '>0.9', '>0.9', '>0.9'],
    })

    cov_rows = params[params['op'] == '~~'].copy()
    cov_cross = cov_rows[
        (cov_rows['lval'].isin(factor_list)) &
        (cov_rows['rval'].isin(factor_list)) &
        (cov_rows['lval'] != cov_rows['rval'])
    ].copy()
    if cov_cross.empty:
        cov_cross = cov_rows[
            (cov_rows['rval'].isin(factor_list)) &
            (cov_rows['lval'].isin(factor_list)) &
            (cov_rows['lval'] != cov_rows['rval'])
        ].copy()

    if len(cov_cross) > 0:
        def _std_cov(row):
            sd1 = factor_sd.get(row['lval'], 1.0)
            sd2 = factor_sd.get(row['rval'], 1.0)
            return row['Estimate'] / (sd1 * sd2)

        cov_cross['标准估计系数'] = cov_cross.apply(_std_cov, axis=1)

        if factor_names:
            cov_cross['lval'] = cov_cross['lval'].map(lambda x: factor_names.get(alias_to_factor.get(x, x), alias_to_factor.get(x, x)))
            cov_cross['rval'] = cov_cross['rval'].map(lambda x: factor_names.get(alias_to_factor.get(x, x), alias_to_factor.get(x, x)))
        else:
            cov_cross['lval'] = cov_cross['lval'].map(lambda x: alias_to_factor.get(x, x))
            cov_cross['rval'] = cov_cross['rval'].map(lambda x: alias_to_factor.get(x, x))

        cov_table = cov_cross.rename(columns={
            'lval': '变量1',
            'rval': '变量2',
            'Estimate': '非标准估计系数',
            'Std. Err': '标准误',
            'z-value': 'Z值',
            'p-value': 'P值',
        })[['变量1', '变量2', '非标准估计系数', '标准误', 'Z值', 'P值', '标准估计系数']].reset_index(drop=True)
    else:
        cov_table = pd.DataFrame()

    factor_summary_rows = []
    for f, items in factors.items():
        display_name = factor_names.get(f, f) if factor_names else f
        factor_summary_rows.append({'Factor': display_name, '数量': int(len(items))})
    factor_summary_rows.append({'Factor': '汇总', '数量': int(len(all_items))})
    factor_summary_rows.append({'Factor': '分析样本量', '数量': int(len(analysis_data))})
    factor_summary_table = pd.DataFrame(factor_summary_rows)

    meta = {
        'method': 'Confirmatory Factor Analysis',
        'analysis_sample_size': int(len(analysis_data)),
        'factor_count': int(len(factors)),
        'item_count': int(len(all_items)),
        'factors': factors,
        'factor_names': factor_names or {},
        'table_keys': ['factor_summary_table', 'load_table', 'eval_table', 'corr_table', 'fit_table', 'cov_table']
    }

    tables = {
        'load_table': load_table,
        'eval_table': eval_table,
        'corr_table': corr_table,
        'fit_table': fit_table,
        'cov_table': cov_table,
        'factor_summary_table': factor_summary_table,
    }

    display = {
        'tables': {
            'load_table': {
                'title': '因子载荷系数表',
                'merge_cells': [
                    {'field': '因子', 'merge_consecutive_same_values': True}
                ],
                'orientation': 'grouped',
                'notes': ['按“因子”列纵向合并相邻相同单元格。']
            },
            'eval_table': {
                'title': '模型评价表（AVE & CR）',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，无需合并单元格。']
            },
            'corr_table': {
                'title': '因子相关矩阵（对角线 = √AVE）',
                'merge_cells': [],
                'orientation': 'matrix',
                'notes': ['首列“因子”为行变量名，其余列为矩阵列。']
            },
            'fit_table': {
                'title': '模型拟合指标表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，直接逐行展示指标即可。']
            },
            'cov_table': {
                'title': '因子协方差表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['如为空表，前端可不展示该表。']
            },
            'factor_summary_table': {
                'title': '因子汇总表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['按因子逐行展示题项数量，并追加汇总与分析样本量。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': tables,
        'display': display,
    }


# ──────────────────────────────────────────────
# 打印结果（便于直接查看）
# ──────────────────────────────────────────────

def print_cfa_results(results: dict):
    """格式化打印 CFA 所有结果表格"""
    sep = "=" * 60

    print(f"\n{sep}")
    print("【1】因子载荷系数表")
    print(sep)
    print(results['tables']['load_table'].to_string(index=False))

    print(f"\n{sep}")
    print("【2】模型评价表（AVE & CR）")
    print(sep)
    print(results['tables']['eval_table'].to_string(index=False))

    print(f"\n{sep}")
    print("【3】因子相关矩阵（对角线 = √AVE）")
    print(sep)
    corr_to_print = results['tables']['corr_table']
    print(corr_to_print.to_string(index=False))

    print(f"\n{sep}")
    print("【4】模型拟合指标")
    print(sep)
    print(results['tables']['fit_table'].to_string(index=False))

    if not results['tables']['cov_table'].empty:
        print(f"\n{sep}")
        print("【5】因子协方差表")
        print(sep)
        print(results['tables']['cov_table'].to_string(index=False))


# ──────────────────────────────────────────────
# 主程序（直接运行时执行示例）
# ──────────────────────────────────────────────

if __name__ == '__main__':
    DATA_PATH = 'your_data.xlsx'
    df = pd.read_excel(DATA_PATH)

    FACTORS = {
        'F1': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        'F2': ['Q11', 'Q12', 'Q13', 'Q14', 'Q15'],
    }

    FACTOR_NAMES = {
        'F1': 'F1',
        'F2': 'F2',
    }

    results = confirmatory_factor_analysis(
        data=df,
        factors=FACTORS,
        factor_names=FACTOR_NAMES,
    )

    print_cfa_results(results)
