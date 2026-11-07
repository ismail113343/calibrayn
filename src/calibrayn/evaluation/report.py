import csv
import json
from dataclasses import asdict
from pathlib import Path

DIGITS = 4


def comparison_rows(comparisons):
    rows = []
    for comparison in comparisons:
        for metric in comparison.metrics:
            rows.append(
                {
                    "persona": comparison.persona,
                    "metric": metric.metric,
                    "plain": round(metric.plain.mean, DIGITS),
                    "adjusted": round(metric.adjusted.mean, DIGITS),
                    "difference": round(metric.difference.mean, DIGITS),
                    "low": round(metric.difference.low, DIGITS),
                    "high": round(metric.difference.high, DIGITS),
                }
            )
    return rows


def curve_rows(comparisons):
    rows = []
    for comparison in comparisons:
        pairs = zip(comparison.plain_curve, comparison.adjusted_curve, strict=True)
        for week, (plain, adjusted) in enumerate(pairs, start=1):
            rows.append(
                {
                    "persona": comparison.persona,
                    "week": week,
                    "plain": round(plain, DIGITS),
                    "adjusted": round(adjusted, DIGITS),
                }
            )
    return rows


def sensitivity_rows(rows, persona):
    return [
        {
            "persona": persona,
            "parameter": row.parameter,
            "value": row.value,
            "retention": round(row.retention.mean, DIGITS),
            "difference": round(row.difference.mean, DIGITS),
            "low": round(row.difference.low, DIGITS),
            "high": round(row.difference.high, DIGITS),
        }
        for row in rows
    ]


def markdown_table(rows):
    if not rows:
        return ""
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_table(folder, name, rows):
    base = Path(folder)
    base.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(base / f"{name}.csv", rows)
    markdown_path = base / f"{name}.md"
    markdown_path.write_text(markdown_table(rows), encoding="utf-8")
    return [csv_path, markdown_path]


def write_comparison(comparisons, folder):
    written = write_table(folder, "persona_comparison", comparison_rows(comparisons))
    written += write_table(folder, "retention_curves", curve_rows(comparisons))
    return written


def write_sensitivity(rows, persona, folder):
    return write_table(folder, f"sensitivity_{rows[0].parameter}", sensitivity_rows(rows, persona))


def write_retrieval(scores, folder):
    base = Path(folder)
    base.mkdir(parents=True, exist_ok=True)
    payload = asdict(scores)
    json_path = base / "retrieval.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table = [{"measure": f"hit@{k}", "value": round(v, DIGITS)} for k, v in scores.hit_at.items()]
    table.append({"measure": "mrr", "value": round(scores.mrr, DIGITS)})
    table.append({"measure": "misses", "value": scores.misses})
    markdown_path = base / "retrieval.md"
    markdown_path.write_text(markdown_table(table), encoding="utf-8")
    return [json_path, markdown_path]
