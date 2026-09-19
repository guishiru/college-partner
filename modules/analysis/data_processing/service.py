"""Safe, traceable CSV/Excel loading and preparation.

This module never overwrites the uploaded file. It returns a new analysis view
and a quality report describing every automatic normalization. Destructive or
ambiguous changes are reported as blocking issues instead of being guessed.
"""

from __future__ import annotations

import csv
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

    try:
        data = pd.read_excel(file_path)
    except Exception as exc:
        raise DataProcessingError(f"Excel 文件读取失败：{exc}") from exc
    if data.empty and len(data.columns) == 0:
        raise DataProcessingError("Excel 文件没有可读取的数据表。")
    return data, {"file_type": suffix[1:], "encoding": None, "delimiter": None}


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

    data = raw.copy(deep=True)
    _normalize_column_names(data, report)
    _normalize_values(data, report)
    _classify_columns(data, report, allow_numeric_text_cast)

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
