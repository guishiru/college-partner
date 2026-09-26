"""
moderation_analysis.py
=======================
调节效应检验 - 支持单个或多个调节变量的调节效应分析

安装依赖：
    python3.11 -m pip install statsmodels openpyxl pandas numpy scipy

快速调用示例：
    import pandas as pd

    from moderation_analysis import moderation_analysis

    df = pd.read_excel('your_data.xlsx')

    results = moderation_analysis(
        data=df,
        x='Q1',              # 自变量
        y='Q2',              # 因变量
        m='Q3',              # 调节变量（单个或列表）
        controls=None,       # 控制变量列表（可选）
    )

    # 返回值是字典，包含：
    # results['model1']              → 主效应模型：Y ~ X
    # results['model2']              → 含交互项模型：Y ~ X + M + X*M
    # results['summary']             → 汇总表
    # results['simple_slope_data']   → 简单斜率图数据表
    # results['simple_slope_config'] → 前端作图配置说明
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ._numeric import NA, numeric_display, significance_star

warnings.filterwarnings('ignore')


def moderation_analysis(data, x, y, m, controls=None, seed=42):
    """
    调节效应分析
    
    Parameters
    ----------
    data : pd.DataFrame
        原始数据
    x : str
        自变量列名
    y : str
        因变量列名
    m : str or list
        调节变量列名（单个或多个）
    controls : list, optional
        控制变量列名列表
    seed : int
        随机种子
    
    Returns
    -------
    dict
        包含模型结果、汇总表和前端作图数据
    """
    np.random.seed(seed)
    
    # 处理调节变量
    if isinstance(m, str):
        m_vars = [m]
    else:
        m_vars = list(m) if m else []
    
    # 准备数据
    all_cols = [x, y] + m_vars + (controls if controls else [])
    df_clean = data[all_cols].dropna().reset_index(drop=True)
    
    n = len(df_clean)
    
    # 使用原始变量（SPSS 口径）
    x_std = x
    y_var = y
    m_std_vars = m_vars
    
    # ═══════════════════════════════════════════════
    # 不使用公式解析，直接用数组方式构建模型
    # ═══════════════════════════════════════════════
    
    # 1. 创建辅助 DataFrame，列名用简单的临时名字
    #    这样可以避免 patsy 解析复杂列名的问题
    temp_df = pd.DataFrame()
    
    # 变量名映射：原始列名 -> 临时简单名
    var_map = {}
    
    # Y
    temp_df['y'] = df_clean[y].values
    var_map['y'] = y
    
    # X
    temp_df['x'] = df_clean[x].values
    var_map['x'] = x
    
    # M
    for i, m_var in enumerate(m_vars):
        temp_name = f'm{i+1}'
        temp_df[temp_name] = df_clean[m_var].values
        var_map[temp_name] = m_var
    
    # 控制变量
    control_temp_names = []
    if controls:
        for i, ctrl in enumerate(controls):
            temp_name = f'c{i+1}'
            temp_df[temp_name] = df_clean[ctrl].values
            var_map[temp_name] = ctrl
            control_temp_names.append(temp_name)
    
    # 构建交互项
    for i, m_var in enumerate(m_vars):
        temp_name = f'm{i+1}'
        inter_name = f'xm{i+1}'
        temp_df[inter_name] = temp_df['x'] * temp_df[temp_name]
        var_map[inter_name] = f'{x}*{m_var}'
    
    # ═══════════════════════════════════════════════
    # 模型1：仅主效应 Y ~ X (+ controls)
    # ═══════════════════════════════════════════════
    X1 = temp_df[['x'] + control_temp_names]
    X1 = sm.add_constant(X1)
    model1 = sm.OLS(temp_df['y'], X1).fit()
    
    # ═══════════════════════════════════════════════
    # 模型1b：含调节变量主效应 Y ~ X + M (+ controls)
    # ═══════════════════════════════════════════════
    m_temp_names = [f'm{i+1}' for i in range(len(m_vars))]
    X1b = temp_df[['x'] + m_temp_names + control_temp_names]
    X1b = sm.add_constant(X1b)
    model1b = sm.OLS(temp_df['y'], X1b).fit()
    
    # ═══════════════════════════════════════════════
    # 模型2：含交互项 Y ~ X + M + X*M (+ controls)
    # ═══════════════════════════════════════════════
    inter_temp_names = [f'xm{i+1}' for i in range(len(m_vars))]
    X2 = temp_df[['x'] + m_temp_names + inter_temp_names + control_temp_names]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(temp_df['y'], X2).fit()
    
    # ═══════════════════════════════════════════════
    # 构建结果表 - 把临时列名映射回原始列名
    # ═══════════════════════════════════════════════
    
    def _get_var_name(temp_var, var_map_dict):
        if temp_var == 'Intercept':
            return '常数'
        return var_map_dict.get(temp_var, temp_var)
    
    # 模型1结果表
    model1_rows = []
    for var in model1.params.index:
        var_name = _get_var_name(var, var_map)
        
        p_val = model1.pvalues[var]
        star = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
        
        model1_rows.append({
            '因变量': y_var,
            '自变量': var_name,
            '系数': model1.params[var],
            '标准误': model1.bse[var],
            't值': model1.tvalues[var],
            'p值': p_val
        })
    
    model1_df = pd.DataFrame(model1_rows)
    
    # 模型2结果表
    model2_rows = []
    for var in model2.params.index:
        var_name = _get_var_name(var, var_map)
        
        p_val = model2.pvalues[var]
        star = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
        
        model2_rows.append({
            '因变量': y_var,
            '自变量': var_name,
            '系数': model2.params[var],
            '标准误': model2.bse[var],
            't值': model2.tvalues[var],
            'p值': p_val
        })
    
    model2_df = pd.DataFrame(model2_rows)
    
    # 模型1b结果表：Y ~ X + M（含调节变量，不含交互项）
    model1b_rows = []
    for var in model1b.params.index:
        var_name = _get_var_name(var, var_map)
        
        p_val = model1b.pvalues[var]
        
        model1b_rows.append({
            '因变量': y_var,
            '自变量': var_name,
            '系数': model1b.params[var],
            '标准误': model1b.bse[var],
            't值': model1b.tvalues[var],
            'p值': p_val
        })
    
    model1b_df = pd.DataFrame(model1b_rows)
    
    # ══════════════════════════════════════════════════
    # 层级回归 △F 检验（hierarchical regression 标准口径）
    # 模型1：Y ~ X             → △F vs 空模型（即模型本身整体 F）
    # 模型1b：Y ~ X + M        → △F vs 模型1（M 的主效应增量）
    # 模型2：Y ~ X + M + X*M   → △F vs 模型1b（交互项增量）
    # F = ((SSR_reduced - SSR_full) / Δdf) / (SSR_full / df_full)
    # p = stats.f.sf(F, df_num, df_den)  [使用生存函数 sf，等价于 1-cdf，避免精度问题]
    # ══════════════════════════════════════════════════

    def _delta_f(reduced_model, full_model):
        """计算嵌套模型的 △F 及其 p 值。"""
        df_num = reduced_model.df_resid - full_model.df_resid
        df_den = full_model.df_resid
        if df_num <= 0 or df_den <= 0:
            return np.nan, np.nan, int(df_num), int(df_den)
        delta_ssr = reduced_model.ssr - full_model.ssr
        if delta_ssr < 0:
            # 完整模型 SSR 反而更大，理论上不应出现；防御性处理
            return np.nan, np.nan, int(df_num), int(df_den)
        f_val = (delta_ssr / df_num) / (full_model.ssr / df_den)
        p_val = float(stats.f.sf(f_val, df_num, df_den))
        return float(f_val), p_val, int(df_num), int(df_den)

    # 模型1 △F：与"零预测变量空模型"比较 = 模型整体 F 检验
    f1, p1, dfn1, dfd1 = (
        float(model1.fvalue),
        float(model1.f_pvalue),
        int(model1.df_model),
        int(model1.df_resid),
    )

    # 模型1b △F：model1b vs model1（加入调节变量主效应的增量）
    f1b, p1b, dfn1b, dfd1b = _delta_f(model1, model1b)

    # 模型2 △F：model2 vs model1b（加入交互项的增量，调节效应核心检验）
    f2, p2, dfn2, dfd2 = _delta_f(model1b, model2)

    # ── 汇总表 ──────────────────────────────────────
    def _star(p):
        return significance_star(p)


    # F 值曾经被写成 'F(3, 176)=35.378' 这样的字符串，自由度和统计量粘在一起，
    # p 值还被 round 到四位并接上星号。报告模块拿不到可用的数，这里全部拆成数值列：
    # 统计量、两个自由度、p 值各自成列，星号是标记单独一列。
    def _model_row(label, model, r2_change, f_change, p_change, dfn, dfd):
        f_value = float(model.fvalue)
        p_value = float(model.f_pvalue)
        return {
            '模型': label,
            'R²': float(model.rsquared),
            '调整R²': float(model.rsquared_adj),
            'F': f_value,
            '模型自由度': int(model.df_model),
            '残差自由度': int(model.df_resid),
            'P值': p_value,
            '显著性': _star(p_value),
            '△R²': float(r2_change),
            '△F': float(f_change),
            '△F分子自由度': int(dfn) if dfn == dfn else NA,
            '△F分母自由度': int(dfd) if dfd == dfd else NA,
            '△P值': float(p_change),
            '△显著性': _star(p_change),
        }

    summary_rows = [
        _model_row('模型1', model1, model1.rsquared, f1, p1, dfn1, dfd1),
        _model_row('模型1b', model1b,
                   model1b.rsquared - model1.rsquared, f1b, p1b, dfn1b, dfd1b),
        _model_row('模型2', model2,
                   model2.rsquared - model1b.rsquared, f2, p2, dfn2, dfd2),
    ]

    summary_df = pd.DataFrame(summary_rows)

    # ── 简单斜率图数据（仅单调节变量时生成，前端据此作图）────────────
    # 设计目标：只返回结构化数据，不在分析层生成图片文件。
    # 前端可将不同“调节水平”作为多条折线，X 作为横轴，Y_hat 作为纵轴。
    simple_slope_data = pd.DataFrame()
    simple_slope_config = {}
    if len(m_vars) == 1:
        m_var = m_vars[0]
        m_std = m_var  # 原始变量，SPSS 口径
        # 用临时名取值
        b0 = float(model2.params.get('Intercept', 0.0))
        b1 = float(model2.params.get('x', 0.0))
        b2 = float(model2.params.get('m1', 0.0))
        b3 = float(model2.params.get('xm1', 0.0))
        
        # 用原始数据计算
        x_points = np.array([df_clean[x_std].min(), df_clean[x_std].max()])
        m_mean = float(df_clean[m_var].mean())
        m_sd   = float(df_clean[m_var].std())
        moderator_levels = [
            ('低水平 (-1SD)', m_mean - m_sd, '#f59e0b'),
            ('平均值',        m_mean,         '#3b82f6'),
            ('高水平 (+1SD)', m_mean + m_sd,  '#10b981'),
        ]

        slope_rows = []
        for level_name, m_level, color in moderator_levels:
            for point_order, x_val in enumerate(x_points, start=1):
                y_hat = b0 + b1 * x_val + b2 * m_level + b3 * x_val * m_level
                slope_rows.append({
                    '调节水平': level_name,
                    '调节值': round(float(m_level), 4),
                    '颜色': color,
                    '点序': point_order,
                    'X': round(float(x_val), 4),
                    'Y_hat': round(float(y_hat), 4),
                })
        simple_slope_data = pd.DataFrame(slope_rows)
        simple_slope_data['注释'] = simple_slope_data.apply(
            lambda row: f"{row['调节水平']}下，第{int(row['点序'])}个作图点：X={row['X']}, 预测Y={row['Y_hat']}",
            axis=1
        )
        simple_slope_data = simple_slope_data[['调节水平', '调节值', '点序', 'X', 'Y_hat', '注释']]

        simple_slope_config = {
            'chart_type': 'line',
            'title': '简单斜率图',
            'x_field': 'X',
            'y_field': 'Y_hat',
            'series_field': '调节水平',
            'series_order': ['平均值', '高水平 (+1SD)', '低水平 (-1SD)'],
            'color_field': '颜色',
            'point_order_field': '点序',
            'x_label': 'X',
            'y_label': 'Y',
            'note': '前端按调节水平分组绘制三条折线；X 为横轴，Y_hat 为预测因变量值；颜色可直接使用“颜色”列。'
        }

    simple_slope_notes = []
    if not simple_slope_data.empty:
        simple_slope_notes = simple_slope_data['注释'].tolist()

    meta = {
        'method': 'Moderation Analysis',
        'analysis_sample_size': int(n),
        'x': x,
        'y': y,
        'moderators': m_vars,
        'controls': controls or [],
        'table_keys': ['model1_table', 'model1b_table', 'model2_table', 'summary_table']
    }

    tables = {
        'model1_table': model1_df,
        'model1b_table': model1b_df,
        'model2_table': model2_df,
        'summary_table': summary_df,
    }

    display = {
        'tables': {
            'model1_table': {
                'title': '模型1：主效应回归表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，无需合并单元格。']
            },
            'model1b_table': {
                'title': '模型1b：加入调节变量主效应回归表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，无需合并单元格。']
            },
            'model2_table': {
                'title': '模型2：含交互项回归表',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['普通宽表，无需合并单元格。']
            },
            'summary_table': {
                'title': '模型对比分析',
                'merge_cells': [],
                'orientation': 'wide',
                'notes': ['按模型逐行展示 R²、△R²、△F 与显著性。']
            },
        },
        'charts': {
            'simple_slope': simple_slope_config
        },
        'text_blocks': {
            'simple_slope_notes': {
                'title': '简单斜率说明',
                'lines': simple_slope_notes,
                'notes': ['返回纯文本注释，不按表格渲染。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': tables,
        'display': display,
        'text_blocks': {
            'simple_slope_notes': simple_slope_notes
        }
    }


if __name__ == '__main__':
    # 测试示例
    df = pd.DataFrame({
        'X': np.random.randn(100),
        'M': np.random.randn(100),
        'Y': np.random.randn(100)
    })
    
    results = moderation_analysis(df, x='X', y='Y', m='M')
    print(results['tables']['summary_table'])
