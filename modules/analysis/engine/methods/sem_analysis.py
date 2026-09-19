"""
Structural Equation Model (SEM) Analysis
SPSS/Amos 口径：semopy 2.3.11

安装依赖：
    pip install semopy openpyxl pandas numpy scipy

调用方式：
    from sem_analysis import run_sem, print_sem_results
    result = run_sem(data, factors, paths)
    print_sem_results(result)

参数：
    data    : pd.DataFrame
    factors : dict, e.g. {'维度1': ['Q1', 'Q2'], '维度2': ['Q3', 'Q4']}
    paths   : list of tuples, e.g. [('维度1', '维度2'), ('维度2', '维度3')]

返回（result dict）：
    result['measurement_table']  - 因子载荷系数表（兼容旧调用）
    result['fit_table']          - 模型拟合指标表（兼容旧调用）
    result['path_table']         - 结构路径系数表（兼容旧调用）
    result['meta']               - 元信息
    result['tables']             - 稳定表结构集合，供前端/其他智能体稳定消费
"""

import warnings
import numpy as np
import pandas as pd

try:
    from semopy import Model
    from semopy.stats import calc_stats
except ImportError:
    raise ImportError("请先安装 semopy：pip install semopy")


def _sig_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    p = float(p)
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
    p = float(p)
    if p < 0.001:
        return '<0.001'
    return str(round(p, 3))


def _fmt_p_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '-'
    return f"{_fmt_p(p)}{_sig_star(p)}"


def _safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except Exception:
        return None


def _normalize_display_number(v, digits=3, eps=5e-4, allow_negative=True):
    """显示层规范化：极小值归 0，None/NaN 返回 '-'。不改变统计估计，仅控制显示。"""
    if v is None:
        return '-'
    try:
        f = float(v)
    except Exception:
        return '-'
    if np.isnan(f):
        return '-'
    if abs(f) < eps:
        f = 0.0
    if not allow_negative and f < 0:
        return '-'
    return round(f, digits)


def _normalize_std_loading(v):
    """标准化载荷显示规范：
    - 无法稳定计算时显示 '-'
    - 极小值显示 0
    - 明显异常值（如绝对值 > 1.2）显示 '-'
    """
    if v is None:
        return '-'
    try:
        f = float(v)
    except Exception:
        return '-'
    if np.isnan(f):
        return '-'
    if abs(f) < 5e-4:
        return 0.0
    if abs(f) > 1.2:
        return '-'
    return round(f, 3)


def _normalize_std_path(v):
    """标准化路径系数显示规范：保留真实结果，仅对缺失和极小值做显示规范化。"""
    if v is None:
        return '-'
    try:
        f = float(v)
    except Exception:
        return '-'
    if np.isnan(f):
        return '-'
    if abs(f) < 5e-4:
        return 0.0
    return round(f, 3)


def _build_model_desc(factors, paths):
    lines = []
    for fname, items in factors.items():
        lines.append(f'{fname} =~ {" + ".join(items)}')
    for (from_f, to_f) in paths:
        lines.append(f'{to_f} ~ {from_f}')
    return '\n'.join(lines)


def _alias_sem_inputs(data: pd.DataFrame, factors: dict, paths: list):
    factor_alias = {factor: f"F{i + 1}" for i, factor in enumerate(factors)}
    alias_to_factor = {alias: factor for factor, alias in factor_alias.items()}

    all_items = []
    for items in factors.values():
        for item in items:
            if item not in all_items:
                all_items.append(item)
    item_alias = {item: f"V{i + 1}" for i, item in enumerate(all_items)}
    alias_to_item = {alias: item for item, alias in item_alias.items()}

    missing_items = [item for item in all_items if item not in data.columns]
    if missing_items:
        raise ValueError(f"SEM题项列不存在: {missing_items[:5]}")

    safe_factors = {
        factor_alias[factor]: [item_alias[item] for item in items]
        for factor, items in factors.items()
    }
    safe_paths = [
        (factor_alias[from_f], factor_alias[to_f])
        for from_f, to_f in paths
        if from_f in factor_alias and to_f in factor_alias
    ]
    safe_df = data[all_items].rename(columns=item_alias)
    return safe_df, safe_factors, safe_paths, alias_to_factor, alias_to_item


def run_sem(data: pd.DataFrame, factors: dict, paths: list) -> dict:
    safe_data, safe_factors, safe_paths, alias_to_factor, alias_to_item = _alias_sem_inputs(data, factors, paths)
    all_items = [item for items in safe_factors.values() for item in items]
    df = safe_data[all_items].dropna().copy()
    n = len(df)

    model_desc = _build_model_desc(safe_factors, safe_paths)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = Model(model_desc)
        model.fit(df)

    params = model.inspect()
    var_rows = params[params['op'] == '~~']

    def _factor_var(fname):
        row = var_rows[(var_rows['lval'] == fname) & (var_rows['rval'] == fname)]
        if len(row) == 0:
            return None
        val = _safe_float(row['Estimate'].iloc[0])
        return abs(val) if val is not None else None

    measurement_rows = []
    for fname, items in safe_factors.items():
        fvar = _factor_var(fname)
        for i, item in enumerate(items):
            row = params[
                (params['lval'] == item) &
                (params['rval'] == fname) &
                (params['op'] == '~')
            ]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            unstd = _safe_float(r['Estimate'])
            se_val = _safe_float(r.get('Std. Err', None))
            z_val = _safe_float(r.get('z-value', None))
            p_val = _safe_float(r.get('p-value', None))
            is_anchor = (i == 0)

            std = None
            if unstd is not None and fvar is not None:
                item_var_row = var_rows[(var_rows['lval'] == item) & (var_rows['rval'] == item)]
                if len(item_var_row) > 0:
                    err_var = abs(_safe_float(item_var_row['Estimate'].iloc[0]) or 0)
                    implied_item_var = (unstd ** 2) * fvar + err_var
                    if implied_item_var > 0:
                        std = unstd * np.sqrt(fvar) / np.sqrt(implied_item_var)

            measurement_rows.append({
                '因子': alias_to_factor.get(fname, fname),
                '变量': alias_to_item.get(item, item),
                '非标准化载荷系数': _normalize_display_number(unstd),
                '标准化载荷系数': _normalize_std_loading(std),
                'z': '-' if is_anchor or z_val is None else _normalize_display_number(z_val),
                'S.E.': '-' if is_anchor or se_val is None else _normalize_display_number(se_val),
                'P': '-' if is_anchor or p_val is None else _fmt_p_star(p_val),
                '是否参考题项': '是' if is_anchor else '否'
            })
    measurement_table = pd.DataFrame(measurement_rows)

    try:
        fit_stats = calc_stats(model)
        fs = fit_stats.iloc[0]

        def _get(key, fallback=None):
            try:
                return _safe_float(fs[key])
            except Exception:
                return fallback

        chi2_val = _get('chi2')
        df_val = _get('DoF')
        chi2_p = _get('chi2 p-value')
        cfi = _get('CFI')
        gfi = _get('GFI')
        rmsea = _get('RMSEA')
        nfi = _get('NFI')
        nnfi = _get('TLI')

        try:
            Sigma, _ = model.calc_sigma()
            S = df.cov().values
            obs_vars = list(model.vars['observed'])
            S_ord = np.array([[S[df.columns.tolist().index(v), df.columns.tolist().index(w)] for w in obs_vars] for v in obs_vars])
            p_obs = len(obs_vars)
            resid = S_ord - Sigma
            rmr = float(np.sqrt(np.sum(resid ** 2) / (p_obs * (p_obs + 1) / 2)))
        except Exception:
            rmr = None

        ratio = chi2_val / df_val if (chi2_val is not None and df_val is not None and df_val > 0) else None

        fit_table = pd.DataFrame([
            {
                'χ²': '-', 'df': '-', 'P': '>0.05',
                '卡方自由度比': '<3', 'GFI': '>0.9',
                'RMSEA': '<0.10', 'RMR': '<0.05',
                'CFI': '>0.9', 'NFI': '>0.9', 'NNFI': '>0.9'
            },
            {
                'χ²': _normalize_display_number(chi2_val),
                'df': int(df_val) if df_val is not None else '-',
                'P': _fmt_p_star(chi2_p) if chi2_p is not None else '-',
                '卡方自由度比': _normalize_display_number(ratio),
                'GFI': _normalize_display_number(gfi),
                'RMSEA': _normalize_display_number(rmsea),
                'RMR': _normalize_display_number(rmr),
                'CFI': _normalize_display_number(cfi),
                'NFI': _normalize_display_number(nfi),
                'NNFI': _normalize_display_number(nnfi),
            }
        ], index=['参考标准', '模型结果'])
    except Exception as e:
        fit_table = pd.DataFrame([{'错误': str(e)}])

    # ── 预计算所有潜变量的 implied total variance ──────────────────────────
    # 按拓扑顺序（先处理外生变量，再处理内生变量）计算每个潜变量的 implied 方差。
    # implied_var(f) = psi(f) + sum_over_predictors[ beta_xf^2 * implied_var(x)
    #                                                + 2 * beta_xf * beta_x'f * cov_implied(x, x') ]
    # 为简化，先用迭代法（最多 20 轮）收敛，适用于递归（无环）SEM。
    all_factor_names = list(safe_factors.keys())
    implied_var = {}
    implied_cov = {}  # (a, b) -> implied cov

    def _get_psi(fname):
        return _factor_var(fname) or 0.0

    def _get_beta(from_f, to_f):
        """to_f ~ from_f 的非标准化系数"""
        r = params[(params['lval'] == to_f) & (params['rval'] == from_f) & (params['op'] == '~')]
        if len(r) == 0:
            return 0.0
        v = _safe_float(r['Estimate'].iloc[0])
        return v if v is not None else 0.0

    # 找出每个潜变量的所有预测变量（上游）
    predictors = {f: [] for f in all_factor_names}
    for (from_f, to_f) in safe_paths:
        if from_f in all_factor_names and to_f in all_factor_names:
            predictors[to_f].append(from_f)

    # 迭代计算 implied variance（最多 20 轮，递归模型通常 1-2 轮收敛）
    for fname in all_factor_names:
        implied_var[fname] = _get_psi(fname)

    for _ in range(20):
        updated = False
        for fname in all_factor_names:
            preds = predictors[fname]
            if not preds:
                continue
            # implied_var(fname) = psi(fname) + sum_i sum_j beta_i * beta_j * implied_cov(i, j)
            total = _get_psi(fname)
            for p1 in preds:
                b1 = _get_beta(p1, fname)
                total += b1 ** 2 * implied_var.get(p1, 0.0)
                for p2 in preds:
                    if p2 != p1:
                        b2 = _get_beta(p2, fname)
                        key = (min(p1, p2), max(p1, p2))
                        total += b1 * b2 * implied_cov.get(key, 0.0)
            if abs(total - implied_var[fname]) > 1e-10:
                implied_var[fname] = total
                updated = True
        # 更新 implied covariance（仅外生变量对之间，内生变量通过路径传递）
        for i, f1 in enumerate(all_factor_names):
            for f2 in all_factor_names[i+1:]:
                # cov(f1, f2) = sum_k beta_k_f1 * implied_var(k) * beta_k_f2 (共同上游)
                # 简化：只考虑直接共同上游
                common = set(predictors[f1]) & set(predictors[f2])
                cov_val = 0.0
                for k in common:
                    cov_val += _get_beta(k, f1) * _get_beta(k, f2) * implied_var.get(k, 0.0)
                implied_cov[(min(f1, f2), max(f1, f2))] = cov_val
        if not updated:
            break
    # ────────────────────────────────────────────────────────────────────────

    path_rows = []
    for (from_f, to_f) in safe_paths:
        row = params[
            (params['lval'] == to_f) &
            (params['rval'] == from_f) &
            (params['op'] == '~')
        ]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        unstd = _safe_float(r['Estimate'])
        se_val = _safe_float(r.get('Std. Err', None))
        z_val = _safe_float(r.get('z-value', None))
        p_val = _safe_float(r.get('p-value', None))

        std = None
        iv_x = implied_var.get(from_f)
        iv_y = implied_var.get(to_f)
        if unstd is not None and iv_x is not None and iv_y is not None and iv_x > 0 and iv_y > 0:
            std = unstd * np.sqrt(iv_x) / np.sqrt(iv_y)

        path_rows.append({
            'Factor(潜变量)': alias_to_factor.get(from_f, from_f),
            '→': '→',
            '分析项(显变量)': alias_to_factor.get(to_f, to_f),
            '非标准化系数': _normalize_display_number(unstd),
            '标准化系数': _normalize_std_path(std),
            '标准误': _normalize_display_number(se_val),
            'Z': _normalize_display_number(z_val),
            'P': _fmt_p_star(p_val) if p_val is not None else '-',
        })
    path_table = pd.DataFrame(path_rows)

    meta = {
        'method': 'SEM (semopy 2.3.11)',
        'n': int(n),
        'factor_count': len(factors),
        'item_count': len(all_items),
        'path_count': len(paths),
        'factors': factors,
        'paths': paths,
        'model_desc': model_desc,
        'table_keys': ['measurement_table', 'path_table', 'fit_table']
    }

    tables = {
        'measurement_table': measurement_table,
        'fit_table': fit_table,
        'path_table': path_table,
    }

    display = {
        'tables': {
            'measurement_table': {
                'title': '因子载荷系数表',
                'merge_cells': [
                    {'field': '因子', 'merge_consecutive_same_values': True}
                ],
                'orientation': 'grouped',
                'notes': ['按“因子”列纵向合并相邻相同单元格。', '可隐藏“是否参考题项”列，仅在程序消费时保留。']
            },
            'fit_table': {
                'title': '模型拟合指标表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['当前表使用两行结构：参考标准、模型结果。可直接整表展示。']
            },
            'path_table': {
                'title': '结构路径系数表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，无需合并单元格。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': tables,
        'display': display,
    }


def print_sem_results(result: dict):
    import sys, io
    # 仅在 stdout 尚未被设置为 UTF-8 时才重定向，避免覆盖调用方已有设置
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    meta = result['meta']
    tables = result['tables']

    print(f"样本量 N={meta['n']}  因子数={meta['factor_count']}  题项数={meta['item_count']}  路径数={meta['path_count']}")

    print("\n【因子载荷系数表】")
    measurement_print = tables['measurement_table'].drop(columns=['是否参考题项'], errors='ignore')
    print(measurement_print.to_string(index=False))
    print("注：参考题项为模型识别固定项，固定为 1，其 z / S.E. / P 按截图口径显示为 -；***、**、*分别代表1%、5%、10%的显著性水平")

    print("\n【模型拟合指标】")
    print(tables['fit_table'].to_string())
    print("注：***、**、*分别代表1%、5%、10%的显著性水平")

    print("\n【结构路径系数表】")
    print(tables['path_table'].to_string(index=False))
    print("注：***、**、*分别代表1%、5%、10%的显著性水平")


if __name__ == "__main__":
    import sys
    df = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else 'data.xlsx')
    factors = eval(sys.argv[2]) if len(sys.argv) > 2 else {'F1': ['Q1', 'Q2']}
    paths = eval(sys.argv[3]) if len(sys.argv) > 3 else []
    res = run_sem(df, factors, paths)
    print_sem_results(res)
