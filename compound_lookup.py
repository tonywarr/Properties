"""Look up compound properties from a table where the first row holds
property names and the second column holds compound names.

Matching rules:
    - Exact match (case-insensitive, whitespace-trimmed) is always preferred.
    - If no exact match exists, the closest compound name is used provided
      its similarity score is >= 95%, but the result is flagged as a
      fuzzy/approximate match so the caller knows it wasn't exact.
    - If the best similarity score is < 95%, no compound is returned and
      the result is flagged as "no match".
    - Any property cell that is blank/NaN is flagged as "no data" rather
      than silently returned as empty.

No output from this module should be treated as a validated data source.
Verify any returned value against the original reference/data sheet before
using it in a calculation, design decision, or deliverable.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

FUZZY_MATCH_THRESHOLD = 95.0  # percent
NO_DATA_FLAG = "NO DATA: property is blank for this compound"

# Column-name substrings (case-insensitive) treated as financial/commercial
# data and excluded from lookups by default, per data-handling policy.
FINANCIAL_COLUMN_PATTERNS = ("cost", "price", "£", "$", "€")


def _is_financial_column(column_name: str) -> bool:
    name = str(column_name).casefold()
    return any(pattern in name for pattern in FINANCIAL_COLUMN_PATTERNS)


@dataclass
class LookupResult:
    requested_name: str
    matched_name: str | None
    match_score: float
    match_type: str  # "exact" | "fuzzy" | "no_match"
    warning: str | None
    properties: dict[str, Any] = field(default_factory=dict)


def load_compound_table(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame, keeping column order."""
    file_path = Path(file_path)
    if file_path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def _normalize(text: str) -> str:
    return str(text).strip().casefold()


def _similarity(a: str, b: str) -> float:
    """Similarity ratio between two strings, expressed as a percentage."""
    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() * 100


def _find_best_match(compound_names: pd.Series, query: str) -> tuple[str | None, float, bool]:
    """Return (best_name, score, is_exact) for query against compound_names."""
    normalized_query = _normalize(query)

    for name in compound_names:
        if _normalize(name) == normalized_query:
            return name, 100.0, True

    best_name = None
    best_score = -1.0
    for name in compound_names:
        score = _similarity(query, name)
        if score > best_score:
            best_name, best_score = name, score

    return best_name, best_score, False


def get_compound_properties(
    file_path: str | Path,
    compound_name: str,
    properties: list[str] | None = None,
    compound_column_index: int = 1,
    exclude_financial_columns: bool = True,
) -> LookupResult:
    """Return properties for `compound_name` from the table at `file_path`.

    `compound_column_index` is the zero-based index of the column holding
    compound names (default 1 = second column, per the source layout).
    `properties`, if given, restricts the result to those property/column
    names; otherwise every column except the compound-name column is
    returned.
    `exclude_financial_columns` (default True) drops any column whose name
    looks like cost/price/financial data (see FINANCIAL_COLUMN_PATTERNS)
    before matching against `properties` or returning results.
    """
    df = load_compound_table(file_path)
    compound_col = df.columns[compound_column_index]
    if exclude_financial_columns:
        df = df[[c for c in df.columns if c == compound_col or not _is_financial_column(c)]]

    best_name, score, is_exact = _find_best_match(df[compound_col], compound_name)

    if best_name is None or score < FUZZY_MATCH_THRESHOLD:
        return LookupResult(
            requested_name=compound_name,
            matched_name=None,
            match_score=round(max(score, 0.0), 1),
            match_type="no_match",
            warning=(
                f"NO MATCH: no compound within {FUZZY_MATCH_THRESHOLD:.0f}% "
                f"similarity to '{compound_name}' was found "
                f"(closest was '{best_name}' at {score:.1f}%)."
                if best_name
                else f"NO MATCH: no compound found for '{compound_name}'."
            ),
        )

    row = df[df[compound_col] == best_name].iloc[0]
    property_columns = [c for c in df.columns if c != compound_col]
    if properties:
        missing = [p for p in properties if p not in property_columns]
        if missing:
            raise KeyError(f"Unknown property name(s): {missing}")
        property_columns = properties

    result_properties: dict[str, Any] = {}
    for prop in property_columns:
        value = row[prop]
        result_properties[prop] = NO_DATA_FLAG if pd.isna(value) or str(value).strip() == "" else value

    warning = None
    match_type = "exact"
    if not is_exact:
        match_type = "fuzzy"
        warning = (
            f"WARNING: no exact match for '{compound_name}'. "
            f"Showing closest match '{best_name}' ({score:.1f}% similarity). "
            "Confirm this is the intended compound before use."
        )

    return LookupResult(
        requested_name=compound_name,
        matched_name=best_name,
        match_score=round(score, 1),
        match_type=match_type,
        warning=warning,
        properties=result_properties,
    )


if __name__ == "__main__":
    # Demo using the small example dataset shipped alongside this script.
    demo_file = Path(__file__).with_name("example_data.csv")

    for query in ["Water", "Watr", "Ethanol", "Xyzzyx"]:
        result = get_compound_properties(demo_file, query)
        print(f"\nRequested: {query}")
        print(f"  Match type : {result.match_type} (score {result.match_score}%)")
        if result.warning:
            print(f"  {result.warning}")
        if result.properties:
            for prop, value in result.properties.items():
                print(f"  {prop}: {value}")
