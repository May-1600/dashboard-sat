"""
Theme configuration for the DSAT Dashboard.

This module defines the brand visual identity including:
- Core palette (orange-centric pastel colors)
- Light and dark theme neutrals
- Spacing and layout tokens
- Mantine theme configuration
- Plotly chart templates
"""

# =============================================================================
# CORE PALETTE (Mid-tones)
# =============================================================================
BRAND_COLORS = {
    # Primary brand color (orange) - for active states, buttons, focus rings
    "primary": "#FF9A5B",
    # Positive status (teal) - "Hors récla" series/tags
    "positive": "#7FD1C3",
    # Negative status (raspberry) - "Récla" series/tags
    "negative": "#E07A8E",
    # Warning status (gold)
    "warning": "#F3C969",
    # Secondary accent (lavender) - non-critical UI, hover highlights
    "secondary": "#B9A7F8",
}

# Semantic aliases for clarity
SERIES_COLORS = {
    "hors_recla": BRAND_COLORS["positive"],  # Teal #7FD1C3
    "recla": BRAND_COLORS["negative"],       # Raspberry #E07A8E
    "dsat_global": BRAND_COLORS["primary"],  # Orange #FF9A5B
}

# Delta/trend colors
DELTA_COLORS = {
    "positive": BRAND_COLORS["positive"],    # Teal for positive trends
    "negative": BRAND_COLORS["negative"],    # Raspberry for negative trends
    "warning": BRAND_COLORS["warning"],      # Gold for warning states
    "neutral": "#6B7280",                    # Gray for stable/neutral
}

# =============================================================================
# LIGHT THEME NEUTRALS
# =============================================================================
LIGHT_NEUTRALS = {
    "background": "#F7F9FC",
    "surface": "#FFFFFF",
    "surface_muted": "#F2F5FA",
    "border": "#E5E9F2",
    "gridlines": "#E6EAF2",
    "text_primary": "#1F2937",
    "text_muted": "#5B677A",
}

# =============================================================================
# DARK THEME NEUTRALS
# =============================================================================
DARK_NEUTRALS = {
    "background": "#0E141B",
    "surface": "#131A22",
    "surface_muted": "#0F1720",
    "border": "#253041",
    "gridlines": "#293445",
    "text_primary": "#E6EDF7",
    "text_muted": "#A8B3C2",
}

# =============================================================================
# SPACING TOKENS (in pixels)
# =============================================================================
SPACING = {
    # Card internal padding
    "card_padding": 16,
    # Space between title/value/delta in KPI cards
    "card_internal_gap": 12,
    # Horizontal grid gutters
    "grid_gutter_h": 20,
    # Vertical grid gutters
    "grid_gutter_v": 24,
    # Section spacing (between stacked sections)
    "section_gap": 24,
    # Gap below filter row before content
    "filter_content_gap": 24,
    # Space between filter controls
    "filter_control_gap": 18,
    # Desktop outer padding
    "page_padding_desktop": 24,
    # Mobile outer padding
    "page_padding_mobile": 16,
    # Mobile stacking gaps
    "mobile_stack_gap": 18,
}

# =============================================================================
# LAYOUT TOKENS
# =============================================================================
LAYOUT = {
    "container_max_width": 1280,
    "table_row_height": 42,
}

# =============================================================================
# MANTINE THEME CONFIGURATION
# =============================================================================
MANTINE_THEME = {
    "primaryColor": "orange",
    "colors": {
        "orange": [
            "#FFF5EE",  # 0 - lightest
            "#FFE8D6",  # 1
            "#FFD4B8",  # 2
            "#FFC099",  # 3
            "#FFAC7A",  # 4
            "#FF9A5B",  # 5 - primary
            "#E88A50",  # 6
            "#D17A45",  # 7
            "#BA6A3A",  # 8
            "#A35A2F",  # 9 - darkest
        ],
        "teal": [
            "#E6FAF6",  # 0
            "#C2F0E8",  # 1
            "#9EE6DA",  # 2
            "#7FD1C3",  # 3 - positive
            "#6BC4B5",  # 4
            "#57B7A7",  # 5
            "#4AA899",  # 6
            "#3D998B",  # 7
            "#308A7D",  # 8
            "#237B6F",  # 9
        ],
        "raspberry": [
            "#FDEEF1",  # 0
            "#F9D4DC",  # 1
            "#F5BAC7",  # 2
            "#F1A0B2",  # 3
            "#E07A8E",  # 4 - negative
            "#D4687C",  # 5
            "#C8566A",  # 6
            "#BC4458",  # 7
            "#B03246",  # 8
            "#A42034",  # 9
        ],
        "gold": [
            "#FFFBEB",  # 0
            "#FEF3C7",  # 1
            "#FDE68A",  # 2
            "#F3C969",  # 3 - warning
            "#FBBF24",  # 4
            "#F59E0B",  # 5
            "#D97706",  # 6
            "#B45309",  # 7
            "#92400E",  # 8
            "#78350F",  # 9
        ],
        "lavender": [
            "#F5F3FF",  # 0
            "#EDE9FE",  # 1
            "#DDD6FE",  # 2
            "#C4B5FD",  # 3
            "#B9A7F8",  # 4 - secondary
            "#A78BFA",  # 5
            "#8B5CF6",  # 6
            "#7C3AED",  # 7
            "#6D28D9",  # 8
            "#5B21B6",  # 9
        ],
    },
    "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "headings": {
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    },
    "defaultRadius": "md",
    "components": {
        "Card": {
            "styles": {
                "root": {
                    "backgroundColor": LIGHT_NEUTRALS["surface"],
                }
            }
        },
        "Tabs": {
            "styles": {
                "tab": {
                    "--tab-border-color": "transparent",
                },
                "tabLabel": {
                    "fontWeight": 500,
                },
            }
        },
        "SegmentedControl": {
            "defaultProps": {
                "radius": "md",
            }
        },
    },
}

# =============================================================================
# CHART STYLE HELPERS
# =============================================================================


def get_chart_colors(theme: str = "light") -> dict:
    """
    Get chart-specific colors based on current theme.
    Series colors remain the same across themes; only canvas/grid/text adapt.
    """
    neutrals = LIGHT_NEUTRALS if theme == "light" else DARK_NEUTRALS

    return {
        # Series colors (same for both themes)
        "hors_recla": SERIES_COLORS["hors_recla"],
        "recla": SERIES_COLORS["recla"],
        "dsat_global": SERIES_COLORS["dsat_global"],
        "primary": BRAND_COLORS["primary"],
        "secondary": BRAND_COLORS["secondary"],
        "warning": BRAND_COLORS["warning"],
        # Canvas/background colors (theme-dependent)
        "paper_bgcolor": neutrals["surface"],
        "plot_bgcolor": neutrals["surface"],
        "gridcolor": neutrals["gridlines"],
        "axis_color": neutrals["text_muted"],
        "text_color": neutrals["text_primary"],
        "text_muted": neutrals["text_muted"],
        "border": neutrals["border"],
    }


def get_plotly_template(theme: str = "light") -> dict:
    """
    Returns a Plotly layout template with brand styling.
    Use this as a base for all charts to ensure consistency.
    """
    colors = get_chart_colors(theme)

    return {
        "layout": {
            "font": {
                "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                "color": colors["text_color"],
            },
            "paper_bgcolor": colors["paper_bgcolor"],
            "plot_bgcolor": colors["plot_bgcolor"],
            "xaxis": {
                "gridcolor": colors["gridcolor"],
                "linecolor": colors["border"],
                "tickfont": {"color": colors["text_muted"]},
                "title": {"font": {"color": colors["text_muted"]}},
            },
            "yaxis": {
                "gridcolor": colors["gridcolor"],
                "linecolor": colors["border"],
                "tickfont": {"color": colors["text_muted"]},
                "title": {"font": {"color": colors["text_muted"]}},
            },
            "legend": {
                "font": {"color": colors["text_color"]},
                "bgcolor": "rgba(255,255,255,0.9)" if theme == "light" else "rgba(19,26,34,0.9)",
                "bordercolor": colors["border"],
                "borderwidth": 1,
            },
            "colorway": [
                BRAND_COLORS["primary"],
                BRAND_COLORS["positive"],
                BRAND_COLORS["negative"],
                BRAND_COLORS["warning"],
                BRAND_COLORS["secondary"],
            ],
        }
    }


def get_heatmap_colorscale(metric: str = "dsat") -> list:
    """
    Returns a colorscale for heatmaps based on metric type.
    Uses brand colors with proper contrast.
    """
    if metric in ("dsat_sans", "dsat_total", "dsat"):
        # For DSAT: red (bad) -> yellow (mid) -> green (good)
        return [
            [0, BRAND_COLORS["negative"]],     # Raspberry for low DSAT (bad)
            [0.5, BRAND_COLORS["warning"]],    # Gold for mid
            [1, BRAND_COLORS["positive"]],     # Teal for high DSAT (good)
        ]
    elif metric == "delta":
        # For delta: similar scale but centered on 0
        return [
            [0, BRAND_COLORS["negative"]],
            [0.5, "#F7F9FC"],  # Neutral background
            [1, BRAND_COLORS["positive"]],
        ]
    else:
        # Default sequential
        return [
            [0, LIGHT_NEUTRALS["surface_muted"]],
            [1, BRAND_COLORS["primary"]],
        ]


def get_text_color_for_background(bg_color: str) -> str:
    """
    Returns appropriate text color (dark/light) based on background luminance.
    Used for labels on colored areas (e.g., heatmap cells).
    """
    # Simple luminance calculation
    hex_color = bg_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        # Relative luminance formula
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return LIGHT_NEUTRALS["text_primary"] if luminance > 0.5 else DARK_NEUTRALS["text_primary"]
    return LIGHT_NEUTRALS["text_primary"]


# =============================================================================
# KPI CARD STYLING
# =============================================================================
KPI_CARD_STYLE = {
    "padding": SPACING["card_padding"],
    "internal_gap": SPACING["card_internal_gap"],
    "background": LIGHT_NEUTRALS["surface"],
    "title_color": "dimmed",  # Text muted
    "value_color": "inherit",  # Text primary
    "value_align": "right",
}


# =============================================================================
# TABLE STYLING (AG-Grid)
# =============================================================================
def get_table_style(theme: str = "light") -> dict:
    """Returns AG-Grid styling tokens."""
    neutrals = LIGHT_NEUTRALS if theme == "light" else DARK_NEUTRALS

    return {
        "row_height": LAYOUT["table_row_height"],
        "zebra_color": neutrals["surface_muted"],
        "border_color": neutrals["border"],
        "text_color": neutrals["text_primary"],
        "header_background": neutrals["surface"],
    }


# =============================================================================
# TAG/BADGE COLORS
# =============================================================================
TAG_COLORS = {
    "hors_recla": {
        "color": BRAND_COLORS["positive"],
        "variant": "outline",
    },
    "recla": {
        "color": BRAND_COLORS["negative"],
        "variant": "outline",
    },
    "neutral": {
        "color": LIGHT_NEUTRALS["text_muted"],
        "variant": "light",
    },
}


# =============================================================================
# ACTIVE STATE STYLING
# =============================================================================
ACTIVE_STATE = {
    "underline_color": BRAND_COLORS["primary"],
    "underline_width": 2,
    "focus_ring_color": BRAND_COLORS["primary"],
}


# =============================================================================
# BACKWARD COMPATIBILITY - Export old COLORS dict
# =============================================================================
# This maintains compatibility with existing code while transitioning
COLORS = {
    "primary": BRAND_COLORS["primary"],
    "success": BRAND_COLORS["positive"],
    "warning": BRAND_COLORS["warning"],
    "danger": BRAND_COLORS["negative"],
    "teal": BRAND_COLORS["positive"],
    "purple": BRAND_COLORS["secondary"],
    "orange": BRAND_COLORS["primary"],
    "gray": LIGHT_NEUTRALS["text_muted"],
    "background": LIGHT_NEUTRALS["background"],
}
