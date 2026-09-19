"""
parallel_mediation_analysis.py
===============================
平行中介效应检验 - BCa Bootstrap（SPSS PROCESS Model 4 多中介口径）
依赖：numpy, pandas, scipy

安装依赖：
    python3.11 -m pip install openpyxl pandas numpy scipy

模型示意（以 M1、M2 两个并列中介为例）：
         ┌──(a1)──▶ M1 ──(b1)──┐
    X ───┤                      ├──▶ Y
         └──(a2)──▶ M2 ──(b2)──┘
    X ──(c')───────────────────▶ Y   直接效应（控制所有 M）
    X ──(c)────────────────────▶ Y   总效应（不含 M）
    总间接效应 = Σ(ai × bi)
    各路径间接效应 = ai × bi（单独 Bootstrap 检验）
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats

warnings.filterwarnings('ignore')


def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    n = len(y)
    Xc = np.column_stack([np.ones(n), X])
    k = Xc.shape[1] - 1

    coef = np.linalg.lstsq(Xc, y, rcond=None)[0]
    y_hat = Xc @ coef
    resid = y - y_hat
    sse = np.sum(resid ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - sse / sst
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    mse = sse / (n - k - 1)
    cov_b = mse * np.linalg.pinv(Xc.T @ Xc)
    se = np.sqrt(np.diag(cov_b))

    t_val = coef / se
    p_val = 2 * stats.t.sf(np.abs(t_val), df=n - k - 1)

    f_val = ((sst - sse) / k) / mse
    f_p = stats.f.sf(f_val, k, n - k - 1)

    return {
        'coef': coef,
        'se': se,
        't': t_val,
        'p': p_val,
        'r2': r2,
        'adj_r2': adj_r2,
        'f': f_val,
        'f_p': f_p,
        'n': n,
        'k': k,
    }


def _sig_star(p: float) -> str:
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.10:
        return '*'
    return ''


def _reg_table(reg: dict, var_names: list, y_arr: np.ndarray, X_arrs: list) -> pd.DataFrame:
    sd_y = np.std(y_arr, ddof=1)
    names = ['常数'] + var_names
    rows = []
    for i, name in enumerate(names):
        p_val = reg['p'][i]
        if i == 0:
            beta = '—'
        else:
            sd_x = np.std(X_arrs[i - 1], ddof=1)
            beta = round(reg['coef'][i] * sd_x / sd_y, 4)
        rows.append({
            '变量': name,
            'B（非标准化）': round(reg['coef'][i], 4),
            'Beta（标准化）': beta,
            'SE': round(reg['se'][i], 4),
            't值': round(reg['t'][i], 4),
            'P值': f"{round(p_val, 4)}{_sig_star(p_val)}",
        })
    df = pd.DataFrame(rows)
    df.attrs.update({
        'r2': round(reg['r2'], 4),
        'adj_r2': round(reg['adj_r2'], 4),
        'f': round(reg['f'], 4),
        'f_p': round(reg['f_p'], 4),
        'n': reg['n'],
    })
    return df


def _bca_ci(ab_point: float, ab_boot: np.ndarray, ab_jack: np.ndarray, ci: float) -> tuple:
    z0 = stats.norm.ppf(np.mean(ab_boot < ab_point))
    jm = np.mean(ab_jack)
    num = np.sum((jm - ab_jack) ** 3)
    den = 6 * (np.sum((jm - ab_jack) ** 2) ** 1.5)
    a_hat = num / den if den != 0 else 0.0

    alpha = 1 - ci
    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    p_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
    p_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))
    ci_lo = np.percentile(ab_boot, p_lo * 100)
    ci_hi = np.percentile(ab_boot, p_hi * 100)

    boot_se = np.std(ab_boot, ddof=1)
    z_val = ab_point / boot_se if boot_se > 0 else np.nan
    p_val = 2 * stats.norm.sf(np.abs(z_val)) if not np.isnan(z_val) else np.nan

    return ci_lo, ci_hi, boot_se, z_val, p_val


def parallel_mediation(data: pd.DataFrame,
                        x: str,
                        mediators: list,
                        y: str,
                        controls: list = None,
                        n_boot: int = 1000,
                        ci: float = 0.95,
                        seed: int = 42) -> dict:
    controls = controls or []
    cols = [x] + mediators + [y] + controls
    df = data[cols].dropna().reset_index(drop=True)
    X_arr = df[x].values.astype(float)
    Y_arr = df[y].values.astype(float)
    M_arrs = [df[m].values.astype(float) for m in mediators]
    C_arr = df[controls].values.astype(float) if controls else np.empty((len(df), 0))
    n_m = len(mediators)
    n = len(df)

    X1 = np.column_stack([X_arr, C_arr]) if controls else X_arr.reshape(-1, 1)
    reg_c = _ols(Y_arr, X1)
    c = reg_c['coef'][1]
    c_p = reg_c['p'][1]

    reg_a_list = []
    a_list = []
    a_p_list = []
    for M_arr_i in M_arrs:
        reg_a = _ols(M_arr_i, X1)
        reg_a_list.append(reg_a)
        a_list.append(reg_a['coef'][1])
        a_p_list.append(reg_a['p'][1])

    XM_arr = np.column_stack([X_arr] + M_arrs + ([C_arr] if controls else []))
    reg_full = _ols(Y_arr, XM_arr)
    c_prime = reg_full['coef'][1]
    c_prime_p = reg_full['p'][1]
    b_list = [reg_full['coef'][i + 2] for i in range(n_m)]
    b_p_list = [reg_full['p'][i + 2] for i in range(n_m)]

    ab_list = [a_list[i] * b_list[i] for i in range(n_m)]
    ab_total = sum(ab_list)

    rng = np.random.default_rng(seed)
    ab_boots = np.empty((n_boot, n_m + 1))

    for boot_i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X_arr[idx]
        Mbs = [M_arrs[j][idx] for j in range(n_m)]
        Yb = Y_arr[idx]
        Cb = C_arr[idx] if controls else np.empty((len(idx), 0))

        X1b = np.column_stack([Xb, Cb]) if controls else Xb.reshape(-1, 1)
        a_b_list = [_ols(Mbs[j], X1b)['coef'][1] for j in range(n_m)]
        XMb = np.column_stack([Xb] + Mbs + ([Cb] if controls else []))
        reg_b = _ols(Yb, XMb)
        b_b_list = [reg_b['coef'][j + 2] for j in range(n_m)]

        for j in range(n_m):
            ab_boots[boot_i, j] = a_b_list[j] * b_b_list[j]
        ab_boots[boot_i, n_m] = sum(a_b_list[j] * b_b_list[j] for j in range(n_m))

    ab_jacks = np.empty((n, n_m + 1))
    for jack_i in range(n):
        idx_j = np.delete(np.arange(n), jack_i)
        Xj = X_arr[idx_j]
        Mjs = [M_arrs[jm][idx_j] for jm in range(n_m)]
        Yj = Y_arr[idx_j]
        Cj = C_arr[idx_j] if controls else np.empty((len(idx_j), 0))

        X1j = np.column_stack([Xj, Cj]) if controls else Xj.reshape(-1, 1)
        a_j_list = [_ols(Mjs[jm], X1j)['coef'][1] for jm in range(n_m)]
        XMj = np.column_stack([Xj] + Mjs + ([Cj] if controls else []))
        reg_j = _ols(Yj, XMj)
        b_j_list = [reg_j['coef'][jm + 2] for jm in range(n_m)]

        for jm in range(n_m):
            ab_jacks[jack_i, jm] = a_j_list[jm] * b_j_list[jm]
        ab_jacks[jack_i, n_m] = sum(a_j_list[jm] * b_j_list[jm] for jm in range(n_m))

    ci_pct = int(ci * 100)
    indirect_results = []
    for j in range(n_m):
        ab_pt = ab_list[j]
        ci_lo, ci_hi, boot_se, z_val, p_val = _bca_ci(ab_pt, ab_boots[:, j], ab_jacks[:, j], ci)
        star_p = _sig_star(p_val) if not np.isnan(p_val) else ''
        indirect_results.append({
            'mediator': mediators[j],
            'a': round(a_list[j], 4),
            'a_p': f"{round(a_p_list[j], 4)}{_sig_star(a_p_list[j])}",
            'b': round(b_list[j], 4),
            'b_p': f"{round(b_p_list[j], 4)}{_sig_star(b_p_list[j])}",
            'ab': round(ab_pt, 4),
            'boot_se': round(boot_se, 4),
            'z_val': round(z_val, 4) if not np.isnan(z_val) else '—',
            'p_val': f"{round(p_val, 4)}{star_p}" if not np.isnan(p_val) else '—',
            'ci_lo': round(ci_lo, 4),
            'ci_hi': round(ci_hi, 4),
            'ci_str': f"[{round(ci_lo, 4)}, {round(ci_hi, 4)}]",
            'sig': ci_lo > 0 or ci_hi < 0,
        })

    tot_lo, tot_hi, tot_se, tot_z, tot_p = _bca_ci(ab_total, ab_boots[:, n_m], ab_jacks[:, n_m], ci)
    tot_star = _sig_star(tot_p) if not np.isnan(tot_p) else ''

    summary_rows = []

    for r in indirect_results:
        conclusion = '部分中介作用' if r['sig'] else '部分中介作用不显著'

        summary_rows.append({
            '项': f"{x}-->{r['mediator']}-->{y}",
            'c总效应': round(c, 4),
            'a': r['a'],
            'a(p值)': r['a_p'],
            'b': r['b'],
            'b(p值)': r['b_p'],
            'a*b中介效应': r['ab'],
            'a*b (Boot SE)': r['boot_se'],
            'a*b (Z值)': r['z_val'],
            'a*b (P值)': r['p_val'],
            f'a*b ({ci_pct}%BootCI)': r['ci_str'],
            "c'直接效应": round(c_prime, 4),
            "c'(p值)": f"{round(c_prime_p, 4)}{_sig_star(c_prime_p)}",
            '检验结论': conclusion,
        })

    summary_df = pd.DataFrame(summary_rows)
    display_summary_df = summary_df.copy()

    ctrl_arrs = [df[c].values.astype(float) for c in controls]
    model_c_table = _reg_table(reg_c, [x] + controls, Y_arr, [X_arr] + ctrl_arrs)
    model_a_tables = [
        _reg_table(reg_a_list[j], [x] + controls, M_arrs[j], [X_arr] + ctrl_arrs)
        for j in range(n_m)
    ]
    model_full_table = _reg_table(reg_full, [x] + mediators + controls, Y_arr, [X_arr] + M_arrs + ctrl_arrs)

    tables = {
        'model_c_table': model_c_table,
        'model_a_tables': model_a_tables,
        'model_full_table': model_full_table,
        'summary_table': summary_df,
        'display_summary_table': display_summary_df,
    }

    meta = {
        'method': 'Parallel Mediation Analysis',
        'analysis_sample_size': int(n),
        'x': x,
        'mediators': mediators,
        'y': y,
        'controls': controls,
        'n_boot': int(n_boot),
        'ci': float(ci),
        'table_keys': ['model_c_table', 'model_a_tables', 'model_full_table', 'summary_table', 'display_summary_table']
    }

    display = {
        'tables': {
            'model_c_table': {
                'title': '总效应模型回归表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，无需合并单元格。']
            },
            'model_a_tables': {
                'title': '各中介 a 路径回归表',
                'merge_cells': [],
                'orientation': 'multi_table',
                'notes': ['这是一个表数组，每个中介变量对应一张回归表。']
            },
            'model_full_table': {
                'title': '完整模型回归表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，无需合并单元格。']
            },
            'summary_table': {
                'title': '中介效应汇总表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，可直接展示。']
            },
            'display_summary_table': {
                'title': '前端展示汇总表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['当前与 summary_table 一致，保留作前端稳定消费入口。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': tables,
        'display': display,
    }


def _print_reg(reg_df: pd.DataFrame):
    print(reg_df.to_string(index=False))
    print(f"  R²={reg_df.attrs['r2']}  调整R²={reg_df.attrs['adj_r2']}"
          f"  F={reg_df.attrs['f']}  F(p)={reg_df.attrs['f_p']}"
          f"  N={reg_df.attrs['n']}")


def print_parallel_results(results: dict):
    sep = '=' * 70
    sep2 = '-' * 70
    meta = results['meta']
    x = meta['x']
    mediators = meta['mediators']
    y = meta['y']
    ci_pct = int(meta['ci'] * 100)
    m_str = ' + '.join(mediators)

    print(f"\n{sep}")
    print(f"平行中介效应检验：{x} → [{m_str}] → {y}")
    print(f"Bootstrap {meta['n_boot']} 次  |  {ci_pct}% BCa 偏差校正加速置信区间")
    print(sep)

    print(f"\n{sep2}")
    print(f"【总效应模型】{x} → {y}（不含中介，路径 c）")
    print(sep2)
    _print_reg(results['tables']['model_c_table'])

    for j, m in enumerate(mediators):
        print(f"\n{sep2}")
        print(f"【a 路径 M{j+1}】{x} → {m}")
        print(sep2)
        _print_reg(results['tables']['model_a_tables'][j])

    print(f"\n{sep2}")
    print(f"【完整模型】{x} + {m_str} → {y}（路径 b & c'）")
    print(sep2)
    _print_reg(results['tables']['model_full_table'])

    print(f"\n{sep}")
    print('【中介效应汇总表】')
    print(sep)
    print(results['tables']['display_summary_table'].to_string(index=False))
    print('\n注：*** p<0.01  ** p<0.05  * p<0.10')
    print(sep)


if __name__ == '__main__':
    DATA_PATH = 'your_data.xlsx'
    X_VAR = 'Q1'
    MEDIATORS = ['Q3', 'Q4']
    Y_VAR = 'Q2'
    CONTROLS = []
    N_BOOT = 1000
    CI = 0.95
    SEED = 42

    df = pd.read_excel(DATA_PATH)

    results = parallel_mediation(
        data=df,
        x=X_VAR,
        mediators=MEDIATORS,
        y=Y_VAR,
        controls=CONTROLS,
        n_boot=N_BOOT,
        ci=CI,
        seed=SEED,
    )

    print_parallel_results(results)
