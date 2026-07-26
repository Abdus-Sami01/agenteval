from __future__ import annotations

import html
from typing import Any

from agenteval.calibration import CalibrationReport
from agenteval.stability import StabilityReport
from agenteval.stats import wilson_interval
from agenteval.types import EvalRun, Outcome

STYLE = """
.ae { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; font-size: 14px; line-height: 1.5; }
.ae h2 { font-size: 18px; margin: 0 0 2px; }
.ae h3 { font-size: 14px; margin: 22px 0 8px; text-transform: uppercase; letter-spacing: .06em; opacity: .65; }
.ae .meta { opacity: .65; font-size: 12px; margin-bottom: 18px; font-family: ui-monospace, monospace; }
.ae .cards { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 6px; }
.ae .card { flex: 1 1 120px; border: 1px solid rgba(127,127,127,.28); border-radius: 8px; padding: 10px 12px; }
.ae .card .v { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
.ae .card .k { font-size: 11px; opacity: .6; text-transform: uppercase; letter-spacing: .05em; }
.ae .ci { font-size: 11px; opacity: .6; font-family: ui-monospace, monospace; }
.ae table { border-collapse: collapse; width: 100%; font-size: 13px; }
.ae th, .ae td { text-align: left; padding: 6px 8px; border-bottom: 1px solid rgba(127,127,127,.2); }
.ae th { font-weight: 600; font-size: 11px; text-transform: uppercase; opacity: .6; }
.ae td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: ui-monospace, monospace; }
.ae .bar { background: rgba(127,127,127,.18); border-radius: 3px; height: 9px; min-width: 60px; }
.ae .bar > i { display: block; height: 9px; border-radius: 3px; background: #2d6a4f; }
.ae .pass { color: #2d6a4f; } .ae .fail { color: #9b2226; } .ae .err { color: #9a6700; }
.ae .wrap { overflow-x: auto; }
.ae .note { font-size: 12px; opacity: .7; margin-top: 8px; padding-left: 10px; border-left: 2px solid rgba(127,127,127,.35); }
@media (prefers-color-scheme: dark) { .ae .bar > i { background: #52b788; } .ae .pass { color: #74c69d; } .ae .fail { color: #e5646e; } }
"""


def _card(label: str, value: str, sub: str = "") -> str:
    extra = f'<div class="ci">{html.escape(sub)}</div>' if sub else ""
    return f'<div class="card"><div class="k">{html.escape(label)}</div><div class="v">{html.escape(value)}</div>{extra}</div>'


def _bar(fraction: float, width_px: int = 90) -> str:
    pct = max(0.0, min(1.0, fraction)) * 100
    return f'<span class="bar" style="display:inline-block;width:{width_px}px"><i style="width:{pct:.1f}%"></i></span>'


def run_to_html(
    run: EvalRun,
    show_failures: int = 25,
    calibration_report: CalibrationReport | None = None,
    stability_report: StabilityReport | None = None,
    title: str = "",
) -> str:
    ci = wilson_interval(run.passed, run.passed + run.failed)
    meta = run.metadata
    heading = html.escape(title or f"{meta.suite_name} - {meta.system_name}")

    parts = [
        f'<section class="ae"><style>{STYLE}</style>',
        f"<h2>{heading}</h2>",
        f'<div class="meta">run {html.escape(meta.run_id)} &middot; seed {meta.seed}'
        + (f" &middot; git {html.escape(meta.git_sha)}" if meta.git_sha else "")
        + f" &middot; {html.escape(meta.python_version)}</div>",
        '<div class="cards">',
        _card("tasks", str(len(run))),
        _card("pass rate", f"{run.pass_rate:.1%}", f"95% CI {ci.low:.1%} - {ci.high:.1%}"),
        _card("passed", str(run.passed)),
        _card("failed", str(run.failed)),
        _card("errored", str(run.errored)),
        _card("mean score", f"{run.mean_score:.3f}"),
        _card("wall time", f"{run.total_ms:.0f}ms"),
        "</div>",
    ]

    if run.passed + run.failed < 30:
        parts.append('<div class="note">Fewer than 30 graded tasks, so this interval is wide. '
                     'Treat small differences between systems as unresolved.</div>')

    by_tag = run.by_tag()
    if by_tag:
        rows = []
        for tag in sorted(by_tag):
            graded = [r for r in by_tag[tag] if r.outcome in (Outcome.PASS, Outcome.FAIL)]
            if not graded:
                continue
            passed = sum(1 for r in graded if r.is_pass)
            tag_ci = wilson_interval(passed, len(graded))
            rows.append(
                f"<tr><td>{html.escape(tag)}</td>"
                f'<td>{_bar(tag_ci.point)}</td>'
                f'<td class="num">{tag_ci.point:.1%}</td>'
                f'<td class="num">[{tag_ci.low:.1%}, {tag_ci.high:.1%}]</td>'
                f'<td class="num">{len(graded)}</td></tr>'
            )
        if rows:
            parts += [
                "<h3>By tag</h3>", '<div class="wrap"><table>',
                "<tr><th>tag</th><th></th><th>pass rate</th><th>95% CI</th><th>n</th></tr>",
                *rows, "</table></div>",
            ]

    if stability_report and stability_report.runs > 1:
        s = stability_report
        combined = s.combined_interval()
        naive = s.naive_interval()
        parts += [
            "<h3>Run-to-run stability</h3>",
            '<div class="cards">',
            _card("runs", str(s.runs)),
            _card("spread", f"{s.spread:.1%}"),
            _card("flaky tasks", f"{len(s.flaky)}", f"{s.flake_rate:.0%} of suite"),
            _card("variance-aware CI", f"{combined.point:.1%}", f"{combined.low:.1%} - {combined.high:.1%}"),
            "</div>",
        ]
        if combined.width > naive.width * 1.05:
            parts.append('<div class="note">Nondeterminism widens the interval beyond the '
                         'single-run estimate. A single run would overstate certainty.</div>')

    if calibration_report:
        c = calibration_report
        parts += [
            "<h3>Calibration</h3>",
            '<div class="cards">',
            _card("ECE", f"{c.ece:.4f}"),
            _card("Brier", f"{c.brier:.4f}"),
            _card("stated conf.", f"{c.mean_confidence:.1%}"),
            _card("actual acc.", f"{c.accuracy:.1%}"),
            "</div>",
        ]
        bins = [b for b in c.bins if b.count]
        if bins:
            rows = [
                f'<tr><td class="num">[{b.low:.1f}, {b.high:.1f})</td>'
                f'<td class="num">{b.count}</td>'
                f'<td class="num">{b.mean_confidence:.1%}</td>'
                f'<td class="num">{b.accuracy:.1%}</td>'
                f'<td class="num {"fail" if abs(b.gap) > 0.1 else ""}">{b.gap:+.1%}</td></tr>'
                for b in bins
            ]
            parts += ['<div class="wrap"><table>',
                      "<tr><th>bin</th><th>n</th><th>confidence</th><th>accuracy</th><th>gap</th></tr>",
                      *rows, "</table></div>"]
        if c.overconfident:
            parts.append('<div class="note">Overconfident: stated confidence exceeds measured accuracy.</div>')

    failures = [r for r in run.results if r.outcome != Outcome.PASS]
    if failures:
        rows = []
        for r in failures[:show_failures]:
            reason = r.error or (r.score.detail if r.score else "")
            css = "err" if r.outcome in (Outcome.ERROR, Outcome.TIMEOUT) else "fail"
            rows.append(
                f"<tr><td>{html.escape(r.task_id)}</td>"
                f'<td class="{css}">{html.escape(r.outcome.value)}</td>'
                f"<td>{html.escape(str(reason)[:160])}</td>"
                f'<td class="num">{r.elapsed_ms:.0f}ms</td></tr>'
            )
        parts += [
            f"<h3>Failures ({len(failures)})</h3>", '<div class="wrap"><table>',
            "<tr><th>task</th><th>outcome</th><th>reason</th><th>time</th></tr>",
            *rows, "</table></div>",
        ]
        if len(failures) > show_failures:
            parts.append(f'<div class="note">Showing {show_failures} of {len(failures)} failures.</div>')

    parts.append("</section>")
    return "".join(parts)


def leaderboard_to_html(runs: dict[str, EvalRun], level: float = 0.95, title: str = "Leaderboard") -> str:
    if not runs:
        return f'<section class="ae"><style>{STYLE}</style><h2>{html.escape(title)}</h2><p>No runs.</p></section>'

    entries = []
    for name, run in runs.items():
        ci = wilson_interval(run.passed, run.passed + run.failed, level)
        entries.append((name, run, ci))
    entries.sort(key=lambda e: -e[2].point)

    rows = []
    for name, run, ci in entries:
        rows.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td>{_bar(ci.point)}</td>"
            f'<td class="num">{ci.point:.1%}</td>'
            f'<td class="num">[{ci.low:.1%}, {ci.high:.1%}]</td>'
            f'<td class="num">{run.mean_score:.3f}</td>'
            f'<td class="num">{run.errored}</td>'
            f'<td class="num">{run.total_ms:.0f}ms</td></tr>'
        )

    note = ""
    if len(entries) > 1:
        best, second = entries[0], entries[1]
        if best[2].low <= second[2].high:
            note = (f'<div class="note">{html.escape(best[0])} and {html.escape(second[0])} have '
                    "overlapping confidence intervals, so the ranking between them is not "
                    "statistically established.</div>")
        else:
            note = (f'<div class="note">{html.escape(best[0])} leads {html.escape(second[0])} '
                    "with non-overlapping intervals.</div>")

    return (
        f'<section class="ae"><style>{STYLE}</style><h2>{html.escape(title)}</h2>'
        '<div class="wrap"><table>'
        "<tr><th>system</th><th></th><th>pass rate</th><th>95% CI</th><th>mean</th><th>err</th><th>time</th></tr>"
        + "".join(rows) + "</table></div>" + note + "</section>"
    )


def write_html(content: str, path: str, page_title: str = "agenteval report") -> None:
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(page_title)}</title></head>"
        "<body style='margin:24px;max-width:1000px'>" + content + "</body></html>"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(document)
