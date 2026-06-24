from __future__ import annotations

PRIMARY = "#2563EB"
PRIMARY_LIGHT = "#DBEAFE"
ACCENT = "#F59E0B"
SUCCESS = "#10B981"
DANGER = "#EF4444"
TEXT = "#1F2937"
TEXT_MUTED = "#6B7280"
BACKGROUND = "#F9FAFB"
SURFACE = "#FFFFFF"
BORDER = "#E5E7EB"
SHADOW = "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
FONT_HEADING = "'Georgia', 'Noto Serif SC', 'Times New Roman', serif"
FONT_BODY = "'Segoe UI', 'Noto Sans SC', 'Helvetica Neue', sans-serif"


def inject_custom_css() -> str:
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

        html, body, [class*="css"] {{
            font-family: {FONT_BODY};
            color: {TEXT};
        }}

        h1, h2, h3, h4 {{
            font-family: {FONT_HEADING};
            color: {TEXT};
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        .metric-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1rem;
            box-shadow: {SHADOW};
        }}

        .section-divider {{
            margin-top: 2rem;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid {PRIMARY_LIGHT};
        }}

        .stButton>button {{
            border-radius: 8px;
            font-weight: 600;
        }}

        .stDownloadButton>button {{
            border-radius: 8px;
        }}
    </style>
    """
