# theme.py
"""
Orange-centric pastel brand theme for the DSAT dashboard.
Provides consistent color tokens for light and dark modes.
"""

# Core palette (mid-tones, theme-agnostic)
BRAND_COLORS = {
    'primary': '#FF9A5B',  # Orange (brand/active accents)
    'positive': '#7FD1C3',  # Teal (Hors récla)
    'negative': '#E07A8E',  # Raspberry (Récla)
    'warning': '#F3C969',  # Gold
    'secondary': '#B9A7F8',  # Lavender
}

# Light theme neutrals
LIGHT_NEUTRALS = {
    'background': '#F7F9FC',
    'surface': '#FFFFFF',
    'surface_muted': '#F2F5FA',
    'border': '#E5E9F2',
    'gridlines': '#E6EAF2',
    'text_primary': '#1F2937',
    'text_muted': '#5B677A',
}

# Dark theme neutrals
DARK_NEUTRALS = {
    'background': '#0E141B',
    'surface': '#131A22',
    'surface_muted': '#0F1720',
    'border': '#253041',
    'gridlines': '#293445',
    'text_primary': '#E6EDF7',
    'text_muted': '#A8B3C2',
}


# Plotly template configuration
def get_plotly_template(mode='light'):
    """
    Returns a Plotly template dict for the given mode ('light' or 'dark').
    Ensures consistent chart styling across the dashboard.
    """
    neutrals = LIGHT_NEUTRALS if mode == 'light' else DARK_NEUTRALS

    return {
        'layout': {
            'paper_bgcolor': neutrals['surface'],
            'plot_bgcolor': neutrals['surface'],
            'font': {
                'family': "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                'color': neutrals['text_primary'],
                'size': 13,
            },
            'title': {
                'font': {'size': 16, 'color': neutrals['text_primary']},
            },
            'xaxis': {
                'gridcolor': neutrals['gridlines'],
                'linecolor': neutrals['border'],
                'tickfont': {'color': neutrals['text_muted'], 'size': 12},
                'titlefont': {'color': neutrals['text_muted'], 'size': 13},
            },
            'yaxis': {
                'gridcolor': neutrals['gridlines'],
                'linecolor': neutrals['border'],
                'tickfont': {'color': neutrals['text_muted'], 'size': 12},
                'titlefont': {'color': neutrals['text_muted'], 'size': 13},
            },
            'legend': {
                'font': {'color': neutrals['text_primary'], 'size': 12},
                'bgcolor': 'rgba(0,0,0,0)',
            },
            'colorway': [
                BRAND_COLORS['primary'],
                BRAND_COLORS['positive'],
                BRAND_COLORS['negative'],
                BRAND_COLORS['warning'],
                BRAND_COLORS['secondary'],
            ],
        }
    }


# Mantine theme configuration
MANTINE_THEME = {
    "colorScheme": "light",
    "primaryColor": "orange",
    "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "fontSizes": {
        "xs": 12,
        "sm": 13,
        "md": 14,
        "lg": 16,
        "xl": 18,
    },
    "spacing": {
        "xs": 8,
        "sm": 12,
        "md": 16,
        "lg": 24,
        "xl": 32,
    },
    "radius": {
        "sm": 4,
        "md": 8,
        "lg": 12,
    },
    "colors": {
        "orange": [
            "#FFF4ED",
            "#FFE8D9",
            "#FFD1B3",
            "#FFBA8C",
            "#FFA366",
            "#FF9A5B",  # [5] primary
            "#E6884D",
            "#CC7640",
            "#B36433",
            "#995226",
        ],
    },
}

# Spacing constants
SPACING = {
    'card_padding': 16,
    'grid_gutter_h': 20,
    'grid_gutter_v': 24,
    'section_gap': 24,
    'control_gap': 16,
    'page_padding_desktop': 24,
    'page_padding_mobile': 16,
}
