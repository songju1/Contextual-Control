#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: Optional[str], default: float = float("nan")) -> float:
    try:
        return float(x) if x not in (None, "", "nan", "NaN") else default
    except Exception:
        return default


def to_int(x: Optional[str], default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x) if x not in (None, "", "nan", "NaN") else default
    except Exception:
        return default


def model_label(model: str, memory_dim: Optional[int]) -> str:
    if model == "L":
        return "L"
    if model == "I":
        return "I"
    if model == "M":
        return f"M{memory_dim}"
    return f"{model}{'' if memory_dim is None else memory_dim}"


def passes_filter(
    row: Dict[str, str],
    prior: str,
    phase_filter: str,
    include_models: Tuple[str, ...],
) -> bool:
    row_prior = row.get("prior", "")
    row_phase = row.get("phase_filter", "")
    model = row.get("model", "")
    mem = to_int(row.get("memory_dim"))

    if row_prior != prior:
        return False
    if row_phase != phase_filter:
        return False

    label = model_label(model, mem)
    return label in include_models


def collect_cells(
    rows: List[Dict[str, str]],
    prior: str,
    phase_filter: str,
    include_models: Tuple[str, ...],
) -> Dict[Tuple[str, str, int], Dict[str, str]]:
    """
    key = (model_label, order, switch)
    """
    out: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    for row in rows:
        if not passes_filter(row, prior=prior, phase_filter=phase_filter, include_models=include_models):
            continue

        model = row.get("model", "")
        mem = to_int(row.get("memory_dim"))
        order = row.get("order", "")
        switch = to_int(row.get("switch"), -1)

        key = (model_label(model, mem), order, switch)
        out[key] = row
    return out


def fmt_mean_se(mean_bits: float, se_bits: float, digits: int = 4) -> str:
    return f"{mean_bits:.{digits}f} ± {se_bits:.{digits}f}"


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write("")
            return
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Format analyze_infoI.py summary CSV into a main table for L / I / M16."
    )
    ap.add_argument("summary_csv", type=str, help="Path to *_summary.csv from analyze_infoI.py")
    ap.add_argument(
        "--out_prefix",
        type=str,
        default="results/infoI_main",
        help="Output prefix for CSV/Markdown table",
    )
    ap.add_argument(
        "--prior",
        type=str,
        default="uniform",
        choices=["uniform", "empirical"],
        help="Which prior column to use",
    )
    ap.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "0", "1"],
        help="Which phase_filter rows to use",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="L,I,M16",
        help="Comma-separated model labels to include, e.g. L,I,M16 or L,I,M8,M16,M32,M64",
    )
    ap.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Digits after decimal point",
    )
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv)
    out_prefix = Path(args.out_prefix)
    include_models = tuple(x.strip() for x in args.models.split(",") if x.strip())

    rows = read_csv(summary_csv)
    cellmap = collect_cells(
        rows,
        prior=args.prior,
        phase_filter=args.phase,
        include_models=include_models,
    )

    # main tasks expected in this project
    tasks = [("AB", 25), ("BA", 30)]

    table_rows: List[Dict[str, str]] = []
    for model_name in include_models:
        out_row: Dict[str, str] = {"model": model_name}
        for order, switch in tasks:
            cell = cellmap.get((model_name, order, switch))
            col = f"{order}{switch}"

            if cell is None:
                out_row[col] = "NA"
                out_row[f"{col}_mean_bits"] = ""
                out_row[f"{col}_se_bits"] = ""
                out_row[f"{col}_n_files"] = ""
                continue

            mean_bits = to_float(cell.get("info_mean_bits"))
            se_bits = to_float(cell.get("info_se_bits"))
            n_files = cell.get("n_files", "")

            out_row[col] = fmt_mean_se(mean_bits, se_bits, digits=args.digits)
            out_row[f"{col}_mean_bits"] = f"{mean_bits:.{args.digits}f}"
            out_row[f"{col}_se_bits"] = f"{se_bits:.{args.digits}f}"
            out_row[f"{col}_n_files"] = n_files

        table_rows.append(out_row)

    csv_out = out_prefix.parent / f"{out_prefix.name}_main_table.csv"
    md_out = out_prefix.parent / f"{out_prefix.name}_main_table.md"

    write_csv(csv_out, table_rows)

    # markdown table
    md_lines = []
    md_lines.append(
        f"# Main table for I(C;A|S)\n\n"
        f"- prior: `{args.prior}`\n"
        f"- phase: `{args.phase}`\n"
        f"- models: `{','.join(include_models)}`\n"
    )
    md_lines.append("| Model | AB25 (bits) | BA30 (bits) |")
    md_lines.append("|---|---:|---:|")
    for row in table_rows:
        md_lines.append(f"| {row['model']} | {row['AB25']} | {row['BA30']} |")
    md_lines.append("")
    md_lines.append("Raw numeric columns are available in the CSV output.")

    write_text(md_out, "\n".join(md_lines) + "\n")

    print(f"[DONE] csv : {csv_out}")
    print(f"[DONE] md  : {md_out}")
    print("")
    print("| Model | AB25 (bits) | BA30 (bits) |")
    print("|---|---:|---:|")
    for row in table_rows:
        print(f"| {row['model']} | {row['AB25']} | {row['BA30']} |")


if __name__ == "__main__":
    main()
    
