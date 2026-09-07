"""Chart theme: one palette, applied by role, in both light and dark.

Colours are assigned by the *job* they do, not by series order:

* identity  -> three categorical slots (blue / orange / aqua), fixed order,
               never cycled.  Three is the cap that validates on all pairs for
               colour-vision deficiency, so nothing here needs a fourth hue.
* state     -> a reserved status palette (good / watch / high / critical) used
               only where the colour *means* a condition.  It never doubles as
               a series colour.

Every chart in the app carries a legend and a table view, which is the required
relief for the one slot that sits below 3:1 contrast on the light surface.
"""

from __future__ import annotations

LIGHT = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    "band": "rgba(42,120,214,0.14)",
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

DARK = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "text": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70"],
    "band": "rgba(57,135,229,0.20)",
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

SEVERITY_COLOUR = {
    "ok": "muted",
    "above normal": "series0",
    "watch": "warning",
    "high": "critical",
    "ramp-up": "muted",
}


def palette(mode: str = "light") -> dict:
    return DARK if mode == "dark" else LIGHT


def apply_layout(fig, mode: str = "light", height: int = 380, legend: bool = True):
    """Recessive chrome, hairline grid, generous padding, hover on by default."""
    p = palette(mode)
    fig.update_layout(
        template=None,
        paper_bgcolor=p["surface"],
        plot_bgcolor=p["surface"],
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            color=p["text_secondary"],
            size=13,
        ),
        title=dict(font=dict(color=p["text"], size=15), x=0, xanchor="left", pad=dict(b=12)),
        margin=dict(l=56, r=16, t=48, b=44),
        height=height,
        hovermode="x unified",
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            # Right-aligned so it never collides with the left-aligned title.
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=p["text_secondary"]),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=p["axis"],
        linewidth=1,
        ticks="outside",
        tickcolor=p["axis"],
        tickfont=dict(color=p["muted"], size=12),
        title_font=dict(color=p["muted"], size=12),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=p["grid"],
        gridwidth=1,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color=p["muted"], size=12),
        title_font=dict(color=p["muted"], size=12),
        automargin=True,
    )
    return fig
