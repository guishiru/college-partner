"""
Frequency Analysis
前端稳定输出版
"""

import pandas as pd


def frequency_analysis(df, items=None):
    if items:
        df = df[items].copy()
    df = df.dropna()

    records = []
    for col in df.columns:
        freq = df[col].value_counts().sort_index()
        pct = freq / len(df) * 100
        cum_pct = pct.cumsum()

        for val, count in freq.items():
            records.append({
                '名称': col,
                '选项': val,
                '频数': count,
                '百分比(%)': pct[val],
                '累计百分比(%)': cum_pct[val],
            })
        records.append({
            '名称': col,
            '选项': '合计',
            '频数': int(freq.sum()),
            '百分比(%)': 100.0,
            '累计百分比(%)': 100.0,
        })

    frequency_table = pd.DataFrame(records)

    meta = {
        'method': 'Frequency Analysis',
        'analysis_sample_size': int(len(df)),
        'item_count': int(df.shape[1]),
        'items': df.columns.tolist(),
        'table_keys': ['frequency_table']
    }

    display = {
        'tables': {
            'frequency_table': {
                'title': '频数分析结果',
                'merge_cells': [
                    {'field': '名称', 'merge_consecutive_same_values': True, 'align': 'center', 'vertical_align': 'middle'}
                ],
                'orientation': 'grouped',
                'notes': ['按“名称”列纵向合并相邻相同单元格并居中展示。', '每个变量块末尾追加一行“合计”。']
            }
        }
    }

    return {
        'meta': meta,
        'tables': {'frequency_table': frequency_table},
        'display': display,
    }


if __name__ == '__main__':
    import sys
    df = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else 'data.xlsx')
    items = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    result = frequency_analysis(df, items)
    print(result['tables']['frequency_table'].to_string(index=False))
