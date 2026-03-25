"""
ai_analyst.py — Claude-powered strategy verdict for AlphaDetector results.

Sends the AlphaReport (structured stats) + the figure (as a PNG image) to
Claude Opus and returns a plain-English analysis of whether the strategy has
real alpha, hidden beta, or other risks.

Usage:
    from alphaforge.IndividualAnalysis.AlphaDetection.ai_analyst import ai_verdict

    analysis = ai_verdict(report, fig)   # returns str
    print(analysis)
"""

import base64
import io
import os

import anthropic
import matplotlib.pyplot as plt
import numpy as np

from .alpha_detection import AlphaReport

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a senior quantitative analyst with deep expertise in alpha/beta
decomposition for systematic trading strategies. You receive:
  1. A structured statistical report from an OLS-based alpha detection pipeline.
  2. An image of six diagnostic plots produced by that pipeline.

Your job is to give the trader a clear, direct, honest verdict:
  - Does this strategy have genuine alpha, or are the returns explained by
    market exposure (beta) or volatility timing?
  - Is the edge stable over time and across market regimes?
  - What are the specific risks or red flags?
  - Should the trader keep, modify, or discard this strategy?

Be concise but thorough. Use plain English — avoid jargon where possible.
Structure your response with clear sections. Be honest even if the news is bad.
Do not hedge excessively. Give a definitive recommendation at the end.
"""

# ── Report serialiser ──────────────────────────────────────────────────────────

def _report_to_text(report: AlphaReport) -> str:
    """Convert an AlphaReport to a structured text block for the prompt."""

    def _reg(label, r):
        if r is None:
            return f"  {label}: not enough data\n"
        if np.isnan(r.alpha):
            return f"  {label}: insufficient observations ({r.n_obs})\n"
        note = f"  [{r.notes[0]}]" if r.notes else ""
        return (
            f"  {label}: α={r.alpha*100:+.4f}%/day  β={r.beta:+.4f}  "
            f"t(α)={r.alpha_tstat:+.2f}  R²={r.r_squared:.4f}  "
            f"n={r.n_obs}  verdict={r.verdict}{note}\n"
        )

    lines = [
        f"ALPHA DETECTION REPORT",
        f"Strategy : {report.strategy_name}",
        f"Pair     : {report.pair}",
        f"Overall verdict (majority vote): {report.verdict()}",
        "",
        "── Core regression ──",
        _reg("Overall", report.core),
        "── Stability (early / late split) ──",
        _reg("Early half", report.stability_early),
        _reg("Late half",  report.stability_late),
        "",
        "── Decomposition ──",
        _reg("Overnight",    report.overnight),
        _reg("Intraday",     report.intraday),
        _reg("Vol-adjusted", report.vol_adjusted),
        "",
        "── Stress / regime tests ──",
    ]
    for name, r in report.stress_results.items():
        lines.append(_reg(name, r))

    if not np.isnan(report.residual_sharpe):
        lines += [
            "",
            "── Alpha stream (beta-neutral residuals) ──",
            f"  Annualised Sharpe : {report.residual_sharpe:.3f}",
            f"  Max drawdown      : {report.residual_max_dd*100:.2f}%",
        ]

    return "\n".join(lines)


# ── Main function ──────────────────────────────────────────────────────────────

def ai_verdict(
    report:    AlphaReport,
    fig:       plt.Figure,
    api_key:   str | None = None,
    stream_fn  = None,
) -> str:
    """
    Send the AlphaReport + figure to Claude Opus and return a plain-English
    strategy analysis.

    Args:
        report:    AlphaReport from AlphaDetector.run_all()
        fig:       The matplotlib figure (will be captured as PNG)
        api_key:   Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        stream_fn: Optional callable(text_chunk) for streaming updates to the
                   caller (e.g. to update a UI label). If None, result is
                   returned all at once.

    Returns:
        str — Claude's analysis.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return (
            "ERROR: No API key found.\n"
            "Set the ANTHROPIC_API_KEY environment variable, or pass api_key= to ai_verdict()."
        )

    client = anthropic.Anthropic(api_key=key)

    # ── Capture figure as base64 PNG ──────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.standard_b64encode(buf.read()).decode()

    # ── Build prompt ──────────────────────────────────────────────────────────
    report_text = _report_to_text(report)

    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64,
            },
        },
        {
            "type": "text",
            "text": (
                "Below is the structured statistical output from the alpha detection pipeline. "
                "The image above shows the six diagnostic plots for this strategy.\n\n"
                f"{report_text}\n\n"
                "Please analyse this strategy and give your verdict."
            ),
        },
    ]

    # ── Call Claude with streaming ─────────────────────────────────────────────
    full_text = []
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        for chunk in stream.text_stream:
            full_text.append(chunk)
            if stream_fn is not None:
                stream_fn(chunk)

    return "".join(full_text)
