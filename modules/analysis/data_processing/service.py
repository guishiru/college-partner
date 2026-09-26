"""Safe, traceable CSV/Excel loading and preparation.

This module never overwrites the uploaded file. It returns a new analysis view
and a quality report describing every automatic normalization. Destructive or
ambiguous changes are reported as blocking issues instead of being guessed.
"""

from __future__ import annotations

import csv
import re
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
CSV_DELIMITERS = (",", "\t", ";", "|")


class DataProcessingError(ValueError):
    """Raised when a file cannot safely become an analysis view."""


@dataclass
class QualityIssue:
    code: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataChange:
    code: str
    message: str
    count: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    filename: str
    file_type: str
    encoding: str | None = None
    delimiter: str | None = None
    original_rows: int = 0
    original_columns: int = 0
    prepared_rows: int = 0
    prepared_columns: int = 0
    columns: list[str] = field(default_factory=list)
    column_types: dict[str, str] = field(default_factory=dict)
    missing_counts: dict[str, int] = field(default_factory=dict)
    invalid_counts: dict[str, int] = field(default_factory=dict)
    issues: list[QualityIssue] = field(default_factory=list)
    changes: list[DataChange] = field(default_factory=list)
    ready_for_analysis: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [asdict(item) for item in self.issues]
        payload["changes"] = [asdict(item) for item in self.changes]
        return payload


@dataclass
class PreparedData:
    data: pd.DataFrame
    report: DataQualityReport


def read_data_file(path: str | os.PathLike[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read CSV or Excel without skipping malformed records."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DataProcessingError(
            f"不支持的文件格式：{suffix or '无扩展名'}。支持 CSV、XLSX 和 XLS。"
        )
    if not file_path.is_file():
        raise DataProcessingError(f"文件不存在：{file_path}")
    if file_path.stat().st_size == 0:
        raise DataProcessingError("文件为空，无法进行分析。")

    if suffix == ".csv":
        return _read_csv(file_path)

    # 工作簿常常第一张是「填表说明」，真正的数据在后面。此前只读第一张，
    # 于是说明页被当成全部数据，还报告「可分析」——用户的数据根本没被看到。
    try:
        sheets = pd.read_excel(file_path, sheet_name=None)
    except Exception as exc:
        raise DataProcessingError(f"Excel 文件读取失败：{exc}") from exc
    if not sheets:
        raise DataProcessingError("Excel 文件没有可读取的数据表。")

    ranked = sorted(
        sheets.items(),
        key=lambda item: (len(item[1]) * max(len(item[1].columns), 1), len(item[1].columns)),
        reverse=True,
    )
    sheet_name, data = ranked[0]
    if data.empty and len(data.columns) == 0:
        raise DataProcessingError("Excel 文件没有可读取的数据表。")

    meta = {"file_type": suffix[1:], "encoding": None, "delimiter": None}
    if len(sheets) > 1:
        # 选了哪张、为什么选它，必须让用户看见，不能默默替他决定。
        meta["sheet_choice"] = {
            "selected": str(sheet_name),
            "all_sheets": [
                {"name": str(name), "rows": int(len(frame)), "columns": int(len(frame.columns))}
                for name, frame in sheets.items()
            ],
        }
    return data, meta


def prepare_for_analysis(
    path: str | os.PathLike[str],
    *,
    allow_numeric_text_cast: bool = True,
) -> PreparedData:
    """Read a file and create a traceable, non-destructive analysis view."""

    raw, read_meta = read_data_file(path)
    report = DataQualityReport(
        filename=Path(path).name,
        file_type=read_meta["file_type"],
        encoding=read_meta.get("encoding"),
        delimiter=read_meta.get("delimiter"),
        original_rows=int(len(raw)),
        original_columns=int(len(raw.columns)),
    )
    if read_meta.get("duplicate_headers"):
        report.issues.append(
            QualityIssue(
                "duplicate_column_name",
                "error",
                "原始表头存在重复列名，必须先明确每列的真实含义。",
                {"columns": read_meta["duplicate_headers"]},
            )
        )

    if read_meta.get("sheet_choice"):
        choice = read_meta["sheet_choice"]
        listed = "、".join(
            f"{item['name']}（{item['rows']}行×{item['columns']}列）"
            for item in choice["all_sheets"]
        )
        report.issues.append(
            QualityIssue(
                "multiple_sheets",
                "warning",
                f"工作簿包含多张表（{listed}），已选择数据量最大的「{choice['selected']}」进行分析。"
                f"若分析对象不是这张表，请只保留目标表后重新上传。",
                choice,
            )
        )

    data = raw.copy(deep=True)
    _normalize_column_names(data, report)
    _normalize_values(data, report)
    _classify_columns(data, report, allow_numeric_text_cast)
    _check_header_row(data, report)
    _check_degenerate_columns(data, report)
    _check_numeric_looking_text(data, report)

    report.prepared_rows = int(len(data))
    report.prepared_columns = int(len(data.columns))
    report.columns = [str(column) for column in data.columns]
    report.missing_counts = {
        str(column): int(data[column].isna().sum()) for column in data.columns
    }
    report.ready_for_analysis = not any(
        issue.severity == "error" for issue in report.issues
    )
    return PreparedData(data=data, report=report)


def _read_csv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    encoding = _detect_encoding(path)
    delimiter = _detect_delimiter(path, encoding)
    duplicate_headers = _duplicate_headers(path, encoding, delimiter)
    try:
        data = pd.read_csv(
            path,
            encoding=encoding,
            sep=delimiter,
            engine="python",
            on_bad_lines="error",
            keep_default_na=True,
            dtype=object,
        )
    except Exception as exc:
        raise DataProcessingError(
            f"CSV 读取失败：存在无法解析的行、引号或列数不一致问题。{exc}"
        ) from exc
    if len(data.columns) == 0:
        raise DataProcessingError("CSV 没有可读取的列。")
    return data, {
        "file_type": "csv",
        "encoding": encoding,
        "delimiter": delimiter,
        "duplicate_headers": duplicate_headers,
    }


def _detect_encoding(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in CSV_ENCODINGS:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise DataProcessingError(
        "无法识别 CSV 编码，请将文件另存为 UTF-8 或 GB18030 后重试。"
    )


def _detect_delimiter(path: Path, encoding: str) -> str:
    sample = path.read_text(encoding=encoding, errors="strict")[:64 * 1024]
    try:
        return csv.Sniffer().sniff(
            sample, delimiters="".join(CSV_DELIMITERS)
        ).delimiter
    except csv.Error:
        candidates = {
            item: max(
                (line.count(item) for line in sample.splitlines()[:20]),
                default=0,
            )
            for item in CSV_DELIMITERS
        }
        delimiter, score = max(candidates.items(), key=lambda item: item[1])
        if score == 0:
            # A one-column CSV has no delimiter to detect; comma is the
            # conventional default and still preserves the complete column.
            return ","
        return delimiter


def _duplicate_headers(path: Path, encoding: str, delimiter: str) -> list[str]:
    with path.open("r", encoding=encoding, newline="") as file:
        header = next(csv.reader(file, delimiter=delimiter), [])
    normalized = [
        " ".join(str(value).replace("\ufeff", "").split()).strip()
        for value in header
    ]
    return sorted(
        {column for column in normalized if column and normalized.count(column) > 1}
    )


def _normalize_column_names(data: pd.DataFrame, report: DataQualityReport) -> None:
    original = [str(column) for column in data.columns]
    normalized = [
        " ".join(column.replace("\ufeff", "").split()).strip()
        for column in original
    ]

    empty_positions = [index for index, column in enumerate(normalized) if not column]
    if empty_positions:
        report.issues.append(
            QualityIssue(
                "empty_column_name",
                "error",
                "存在空列名，无法安全匹配变量。",
                {"positions": empty_positions},
            )
        )

    duplicates = sorted(
        {column for column in normalized if normalized.count(column) > 1 and column}
    )
    if duplicates:
        report.issues.append(
            QualityIssue(
                "duplicate_column_name",
                "error",
                "存在重复列名，必须先明确每列的真实含义。",
                {"columns": duplicates},
            )
        )

    if original != normalized:
        report.changes.append(
            DataChange(
                "normalize_column_names",
                "已去除列名首尾空格、BOM 和连续空白。",
                sum(left != right for left, right in zip(original, normalized)),
            )
        )
    data.columns = normalized


def _normalize_values(data: pd.DataFrame, report: DataQualityReport) -> None:
    for column in data.columns:
        series = data[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue
        original = series.copy()
        normalized = series.map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        blank_mask = normalized.map(
            lambda value: isinstance(value, str) and value == ""
        )
        normalized = normalized.mask(blank_mask, pd.NA)
        trim_count = int(
            sum(
                isinstance(left, str)
                and isinstance(right, str)
                and left != right
                for left, right in zip(original, normalized)
            )
        )
        blank_count = int(blank_mask.sum())
        data[column] = normalized
        if trim_count:
            report.changes.append(
                DataChange(
                    "trim_cell_whitespace",
                    f"已去除列“{column}”单元格首尾空格。",
                    trim_count,
                    {"column": column},
                )
            )
        if blank_count:
            report.changes.append(
                DataChange(
                    "blank_to_missing",
                    f"已将列“{column}”中的空字符串标记为缺失。",
                    blank_count,
                    {"column": column},
                )
            )


def _classify_columns(
    data: pd.DataFrame,
    report: DataQualityReport,
    allow_numeric_text_cast: bool,
) -> None:
    for column in data.columns:
        series = data[column]
        non_missing = series.dropna()
        if pd.api.types.is_numeric_dtype(series):
            report.column_types[column] = "numeric"
            non_finite = pd.Series(series).map(
                lambda value: (
                    not pd.isna(value)
                    and isinstance(value, (float, np.floating))
                    and not math.isfinite(float(value))
                )
            )
            invalid = int(non_finite.sum())
            if invalid:
                report.invalid_counts[column] = invalid
                report.issues.append(
                    QualityIssue(
                        "non_finite_value",
                        "error",
                        f"列“{column}”包含无穷值，不能直接分析。",
                        {"column": column, "count": invalid},
                    )
                )
            continue

        if non_missing.empty:
            report.column_types[column] = "unknown"
            report.issues.append(
                QualityIssue(
                    "empty_column",
                    "error",
                    f"列“{column}”没有有效值。",
                    {"column": column},
                )
            )
            continue

        as_text = non_missing.map(str).map(str.strip)
        converted = pd.to_numeric(as_text, errors="coerce")
        numeric_count = int(converted.notna().sum())
        invalid_count = int(converted.isna().sum())
        numeric_like = numeric_count > 0
        if numeric_like and invalid_count == 0 and allow_numeric_text_cast:
            data[column] = pd.to_numeric(
                data[column].map(
                    lambda value: value.strip()
                    if isinstance(value, str)
                    else value
                ),
                errors="coerce",
            )
            report.column_types[column] = "numeric"
            report.changes.append(
                DataChange(
                    "cast_numeric_text",
                    f"已将列“{column}”中明确的数字文本转换为数值。",
                    int(converted.notna().sum()),
                    {"column": column},
                )
            )
            continue

        report.column_types[column] = "categorical"
        invalid_mask = converted.isna()
        if numeric_like and invalid_count:
            report.column_types[column] = "mixed"
            report.invalid_counts[column] = invalid_count
            report.issues.append(
                QualityIssue(
                    "mixed_numeric_text",
                    "error",
                    f"列“{column}”同时包含数字和不能解释为数字的文本，不能安全用于数值分析。",
                    {
                        "column": column,
                        "count": invalid_count,
                        "examples": as_text[invalid_mask].head(5).tolist(),
                    },
                )
            )


# ---------------------------------------------------------------------------
# 静默通过的几类问题
# ---------------------------------------------------------------------------
#
# 这些数据都能被读进来，也都不会让分析崩掉——它们会安安静静地产出一个没有
# 意义的数。按项目约定「所有变更必须可追溯并向用户展示」，这里一律出具警告：
# 不替用户做决定，但也不装作没看见。

_THOUSANDS_PATTERN = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_PERCENT_PATTERN = re.compile(r"^-?\d+(\.\d+)?\s*%$")
_NUMERIC_HEADER_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


def _check_header_row(data: pd.DataFrame, report: DataQualityReport) -> None:
    """列名整片都是数字，多半是首行数据被当成了表头。"""

    columns = [str(column) for column in data.columns]
    if not columns:
        return
    numeric_like = [c for c in columns if _NUMERIC_HEADER_PATTERN.match(c.strip())]
    if len(numeric_like) == len(columns):
        report.issues.append(
            QualityIssue(
                "header_row_looks_like_data",
                "error",
                "所有列名都是数字，首行很可能是数据而不是表头。"
                "若直接分析，这一行样本会被当作列名丢失。请补上表头后重新上传。",
                {"columns": columns},
            )
        )


def _check_degenerate_columns(data: pd.DataFrame, report: DataQualityReport) -> None:
    """全空列和常数列：能算，但算出来的数没有意义。"""

    empty, constant = [], []
    for column in data.columns:
        series = data[column].dropna()
        if series.empty:
            empty.append(str(column))
        elif series.nunique() == 1:
            constant.append(str(column))

    if empty:
        report.issues.append(
            QualityIssue(
                "empty_column",
                "warning",
                f"以下列没有任何有效值：{'、'.join(empty)}。"
                f"把它们纳入分析会因为成对剔除把其他变量的样本量一起清零。",
                {"columns": empty},
            )
        )
    if constant:
        report.issues.append(
            QualityIssue(
                "constant_column",
                "warning",
                f"以下列所有取值都相同（方差为 0）：{'、'.join(constant)}。"
                f"它们无法参与相关、回归、信度等需要变异的分析。",
                {"columns": constant},
            )
        )


def _check_numeric_looking_text(data: pd.DataFrame, report: DataQualityReport) -> None:
    """带千分位或百分号的列：看着是数值，但不能替用户猜。"""

    thousands, percent = [], []
    for column in data.columns:
        series = data[column].dropna().astype(str).str.strip()
        if series.empty:
            continue
        if (series.map(lambda v: bool(_THOUSANDS_PATTERN.match(v))).mean()) > 0.8:
            thousands.append(str(column))
        elif (series.map(lambda v: bool(_PERCENT_PATTERN.match(v))).mean()) > 0.8:
            percent.append(str(column))

    suspects = thousands + percent
    if suspects:
        hints = []
        if thousands:
            hints.append(f"含千分位逗号：{'、'.join(thousands)}")
        if percent:
            hints.append(f"含百分号：{'、'.join(percent)}")
        report.issues.append(
            QualityIssue(
                "numeric_looking_text",
                "warning",
                f"以下列看起来是数值但含有格式符号（{'；'.join(hints)}），"
                f"已按分类变量处理，未自动转换。"
                f"自动转换需要替你判断量纲（例如 22% 是 22 还是 0.22），"
                f"请在源文件中改为纯数字后重新上传。",
                {"thousands_separator": thousands, "percent_sign": percent},
            )
        )
