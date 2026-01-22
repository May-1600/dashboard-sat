from dash import Dash, dcc, html, Input, Output, State
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from .function import *
from .theme import (
    BRAND_COLORS,
    LIGHT_NEUTRALS,
    DARK_NEUTRALS,
    MANTINE_THEME,
    SPACING,
    get_plotly_template,
)


EXCEL_PATH = "djingo_sat_aia_partitioned.xlsx"

df_init = get_data_bouchon()
MAX_YEAR, MAX_MONTH = latest_available_period(df_init)
DEFAULT_MONTH_VALUE = f"{MAX_YEAR}-{MAX_MONTH:02d}-01"
MIN_YEAR, MIN_MONTH = oldest_available_period(df_init)
MIN_MONTH_VALUE = f"{MIN_YEAR}-{MIN_MONTH:02d}-01"


def create_metric_card(title, value_id, subtitle_id=None, icon=None, sparkline_id=None):
    """
    Create a KPI card with neutral background, right-aligned value, and optional sparkline.
    Delta icons use brand colors (teal/raspberry/gold) without background fill.
    """
    sparkline = (
        dcc.Graph(
            id=sparkline_id,
            config={"displayModeBar": False},
            style={
                "height": 50,
                "width": "100%",
                "minWidth": "140px",
                "flexGrow": 1,
            }
        )
        if sparkline_id
        else None
    )

    card_content = [
        dmc.Group(
            [
                dmc.ThemeIcon(
                    DashIconify(icon=icon or "material-symbols:monitoring", width=20),
                    size="lg",
                    radius="md",
                    variant="light",
                    color="orange",
                ),
                dmc.Box(flex=1),
            ],
            justify="space-between",
            mb=SPACING['card_padding'] // 2,
        ),
        dmc.Text(title, size="xs", c="dimmed", fw=500, mb=4),
        dmc.Group(
            [
                dmc.Title(
                    id=value_id,
                    order=3,
                    children="--",
                    style={"lineHeight": 1, "textAlign": "right", "minWidth": "60px"}
                ),
                sparkline,
            ],
            justify="space-between",
            align="center",
            gap="sm",
            wrap="nowrap",
        ),
    ]

    if subtitle_id:
        card_content.append(
            dmc.Group(
                [
                    dmc.Text("Sans récla:", size="xs", c="dimmed"),
                    dmc.Text(
                        id=subtitle_id,
                        size="xs",
                        fw=500,
                        children="--",
                        style={"textAlign": "right"}
                    ),
                ],
                gap=4,
                mt=SPACING['card_padding'] // 2,
                justify="space-between",
            )
        )

    return dmc.Card(
        card_content,
        withBorder=True,
        shadow="sm",
        radius="md",
        p=SPACING['card_padding'],
        style={
            "height": "100%",
            "minWidth": 0,
            "backgroundColor": LIGHT_NEUTRALS['surface'],
        }
    )


dmc.add_figure_templates()

layout = dmc.MantineProvider(
    theme=MANTINE_THEME,
    children=dmc.Container(
        [
            dcc.Store("metrics-store", "memory", {}),
            # Header
            dmc.Stack(
                [
                    dmc.Group(
                        [
                            dmc.Box(
                                [
                                    dmc.Title("Analyse DSAT", order=2, mb=4),
                                    dmc.Text("Suivi de la satisfaction client", size="sm", c="dimmed"),
                                ],
                                flex=1,
                                style={"minWidth": 0}
                            ),
                            dmc.Group(
                                [
                                    dmc.Box(
                                        dmc.MonthPickerInput(
                                            id="month-select",
                                            label="Période",
                                            value=DEFAULT_MONTH_VALUE,
                                            minDate=MIN_MONTH_VALUE,
                                            maxDate=DEFAULT_MONTH_VALUE,
                                            valueFormat="YYYY-MM",
                                            allowDeselect=False,
                                            size="md",
                                            dropdownType="popover",
                                            leftSection=DashIconify(icon="material-symbols:calendar-month", width=20),
                                            style={"width": "100%"}
                                        ),
                                        style={"minWidth": 220, "width": "100%"}
                                    ),
                                ],
                                gap="sm",
                                align="stretch",
                                style={"flex": 1, "minWidth": 220}
                            ),
                        ],
                        justify="space-between",
                        align="flex-start",
                        mb="sm",
                        gap="sm",
                        wrap="wrap"
                    ),

                    # Main Trend Chart
                    dmc.Card(
                        [
                            dmc.Group(
                                [
                                    dmc.Text("Évolution DSAT et volumes sur l'année ...", size="lg", fw=600, id="main-graph-title"),
                                    dmc.Badge("Tendance 12 mois", variant="light", color="gray"),
                                ],
                                justify="space-between",
                                align="center",
                                gap="sm",
                                wrap="wrap"
                            ),
                            dcc.Graph(
                                id="main-graph",
                                config={"displayModeBar": True, "displaylogo": False},
                                style={"height": 320, "width": "100%"},
                                figure=build_main_graph()
                            )
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="lg",
                        p="lg",
                        mb="sm"
                    ),

                    # KPI Cards Row
                    dmc.Grid(
                        [
                            dmc.GridCol(
                                create_metric_card(
                                    "DSAT Global",
                                    "dsat-main-value",
                                    "dsat-sans-value",
                                    icon="material-symbols:sentiment-dissatisfied",
                                    sparkline_id="dsat-spark",
                                    color=COLORS['danger']
                                ),
                                span={"base": 12, "sm": 6, "xl": 3}
                            ),
                            dmc.GridCol(
                                create_metric_card(
                                    "Volume Brut",
                                    "vol-brut-value",
                                    "vol-brut-sans-value",
                                    icon="material-symbols:database",
                                    sparkline_id="vol-brut-spark",
                                    color=COLORS['primary']
                                ),
                                span={"base": 12, "sm": 6, "xl": 3}
                            ),
                            dmc.GridCol(
                                create_metric_card(
                                    "Volume Noté",
                                    "vol-note-value",
                                    "vol-note-sans-value",
                                    icon="material-symbols:star-rate",
                                    sparkline_id="vol-note-spark",
                                    color=COLORS['success']
                                ),
                                span={"base": 12, "sm": 6, "xl": 3}
                            ),
                            dmc.GridCol(
                                dmc.Card(
                                    [
                                        dmc.Group(
                                            [
                                                dmc.ThemeIcon(
                                                    DashIconify(icon="material-symbols:trending-up", width=24),
                                                    size="xl",
                                                    radius="md",
                                                    variant="light",
                                                    id="delta-icon",
                                                    color="blue"
                                                ),
                                            ],
                                            mb="sm"
                                        ),
                                        dmc.Text("Évolution vs M-1", size="sm", c="dimmed", fw=500, mb=4),
                                        dmc.Group(
                                            [
                                                dmc.Title(id="dsat-delta-value", order=2, children="--"),
                                                dmc.Badge(
                                                    id="dsat-delta-badge",
                                                    children="",
                                                    variant="light",
                                                    size="lg",
                                                    radius="md"
                                                ),
                                            ],
                                            justify="space-between",
                                            align="center",
                                            gap="sm",
                                            wrap="wrap"
                                        ),
                                        dmc.Text(
                                            id="delta-trend-text",
                                            size="xs",
                                            c="dimmed",
                                            mt="xs",
                                            children="Variation mensuelle du DSAT"
                                        ),
                                    ],
                                    withBorder=True,
                                    shadow="sm",
                                    radius="lg",
                                    p="lg",
                                    style={"height": "100%", "minWidth": 0}
                                ),
                                span={"base": 12, "sm": 6, "xl": 3}
                            ),
                        ],
                        gutter="lg",
                        mb=0
                    ),

                    # Insight Card - Dynamic based on data
                    dmc.Alert(
                        id="insight-card",
                        title="Analyse en cours...",
                        color="blue",
                        variant="light",
                        icon=DashIconify(icon="material-symbols:lightbulb", width=24),
                        children="Sélectionnez une période pour voir les insights",
                        mb="sm",
                        radius="lg"
                    ),

                    # Analysis Section
                    dmc.Card(
                        [
                            dmc.Group(
                                [
                                    dmc.Text("Analyse Détaillée", size="xl", fw=600),
                                    dmc.Group(
                                        [
                                            dmc.SegmentedControl(
                                                id="grpby-select",
                                                value="pilier",
                                                data=[
                                                    {"label": "Par Pilier", "value": "pilier"},
                                                    {"label": "Par Intention", "value": "intention"},
                                                ],
                                                size="lg",
                                            ),
                                            dmc.Group(
                                                id="intention-controls",
                                                children=[
                                                    dmc.Select(
                                                        id="top-n-select",
                                                        value="10",
                                                        data=[
                                                            {"label": "Top 5", "value": "5"},
                                                            {"label": "Top 10", "value": "10"},
                                                            {"label": "Top 15", "value": "15"},
                                                            {"label": "Top 20", "value": "20"},
                                                            {"label": "Top 30", "value": "30"},
                                                        ],
                                                        size="lg",
                                                        leftSection=DashIconify(icon="material-symbols:filter-list",
                                                                                width=16),
                                                        style={"width": 140},
                                                    ),
                                                    dmc.Tooltip(
                                                        label=(
                                                            "Par défaut, seules les valeurs avec au moins 50 verbatims sont conservées. "
                                                            "Désactiver ce filtrage peut produire un classement basé sur très peu de cas."
                                                        ),
                                                        multiline=True,
                                                        withArrow=True,
                                                        position="bottom",
                                                        children=dmc.Switch(
                                                            id="filter-min50-checkbox",
                                                            label="Filtrage min 50",
                                                            checked=True,
                                                        ),
                                                    ),
                                                ],
                                                gap="md",
                                                align="center",
                                                wrap="wrap",
                                            ),
                                        ],
                                        gap="md",
                                        wrap="wrap"
                                    ),
                                ],
                                justify="space-between",
                                align="center",
                                gap="md",
                                wrap="wrap"
                            ),

                            dmc.Tabs(
                                [
                                    dmc.TabsList(
                                        [
                                            dmc.TabsTab(
                                                dmc.Flex([
                                                    DashIconify(icon="material-symbols:bar-chart", width=18),
                                                    dmc.Text("Classement", size="sm", lh=10)
                                                ], gap=8, align="center"),
                                                value="classement",
                                            ),
                                            dmc.TabsTab(
                                                dmc.Flex([
                                                    DashIconify(icon="material-symbols:scatter-plot", width=18),
                                                    dmc.Text("Priorité", size="sm", lh=10)
                                                ], gap=8, align="center"),
                                                value="quadrants"
                                            ),
                                            dmc.TabsTab(
                                                dmc.Flex([
                                                    DashIconify(icon="material-symbols:calendar-view-month", width=18),
                                                    dmc.Text("Tendance", size="sm", lh=10)
                                                ], gap=8, align="center"),
                                                value="tendance"
                                            ),
                                            dmc.TabsTab(
                                                dmc.Flex([
                                                    DashIconify(icon="material-symbols:account-tree", width=18),
                                                    dmc.Text("Tableur", size="sm", lh=10)
                                                    ], gap=8, align="center",),
                                                value="composition"
                                            ),
                                        ],
                                        grow=False,
                                    ),

                                    dmc.TabsPanel(
                                        dmc.Box(
                                            [
                                                dmc.Text(
                                                    "Classement des catégories par DSAT avec et sans réclamations",
                                                    size="md",
                                                    c="dimmed",
                                                ),
                                                dcc.Graph(
                                                    id="fig-classement",
                                                    config={"displayModeBar": True, "displaylogo": False},
                                                    style={"width": "100%"}
                                                ),
                                            ],
                                            pt="md"
                                        ),
                                        value="classement",
                                    ),
                                    dmc.TabsPanel(
                                        dmc.Box(
                                            [
                                                dmc.Group(
                                                    [
                                                        dmc.Text("Vue agrégée par catégorie", size="md", c="dimmed"),
                                                    ],
                                                    justify="space-between",
                                                    mb="md",
                                                    wrap="wrap"
                                                ),
                                                dmc.Grid(
                                                    children=[
                                                        dmc.GridCol(
                                                            [
                                                                dmc.Skeleton(
                                                                    id="composition-grid-skeleton",
                                                                    visible=False,
                                                                    width="100%",
                                                                    radius="md",
                                                                    children=[
                                                                        dag.AgGrid(
                                                                            id="composition-grid",
                                                                            className="ag-theme-quartz",
                                                                            columnDefs=[],
                                                                            rowData=[],
                                                                            defaultColDef={
                                                                                "sortable": True,
                                                                                "filter": True,
                                                                                "resizable": True,
                                                                                "minWidth": 140,
                                                                            },
                                                                            dashGridOptions={
                                                                                "rowSelection": {"mode": "multiRow"},
                                                                                "animateRows": True,
                                                                            },
                                                                            style={"width": "100%", "height": "420px"},
                                                                        ),
                                                                    ]
                                                                ),
                                                            ],
                                                            span={"base": 12, "lg": 8}
                                                        ),
                                                        dmc.GridCol(
                                                            [
                                                                dmc.Skeleton(
                                                                    id="composition-pie-skeleton",
                                                                    visible=False,
                                                                    width="100%",
                                                                    radius="md",
                                                                    children=[
                                                                        dcc.Graph(
                                                                            id="composition-pie",
                                                                            config={"displayModeBar": False, "displaylogo": False},
                                                                            style={"width": "100%"}
                                                                        ),
                                                                    ]
                                                                ),

                                                            ],
                                                            span={"base": 12, "lg": 4}
                                                        ),
                                                    ],
                                                    gutter="lg"
                                                ),
                                            ],
                                            pt="md",
                                        ),
                                        value="composition",
                                    ),
                                    dmc.TabsPanel(
                                        dmc.Box(
                                            [
                                                dmc.Text(
                                                    "Matrice DSAT vs Volume - Identifiez les zones prioritaires",
                                                    size="sm",
                                                    c="dimmed",
                                                ),
                                                dcc.Graph(
                                                    id="fig-quadrants",
                                                    config={"displayModeBar": True, "displaylogo": False},
                                                    style={"width": "100%"}
                                                ),
                                            ],
                                            pt="md"
                                        ),
                                        value="quadrants",
                                    ),
                                    dmc.TabsPanel(
                                        dmc.Box(
                                            [
                                                dmc.Group(
                                                    [
                                                        dmc.Text(
                                                            "Évolution temporelle par catégorie",
                                                            size="sm",
                                                            c="dimmed"
                                                        ),
                                                        dmc.SegmentedControl(
                                                            id="heatmap-metric",
                                                            value="dsat_total",
                                                            data=[
                                                                {"label": "DSAT sans récla", "value": "dsat_sans"},
                                                                {"label": "Δ (écart)", "value": "delta"},
                                                                {"label": "DSAT total", "value": "dsat_total"},
                                                            ],
                                                            size="xs",
                                                        ),
                                                    ],
                                                    justify="space-between",
                                                    align="center",
                                                    gap="sm",
                                                    wrap="wrap"
                                                ),
                                                dcc.Graph(
                                                    id="fig-tendance",
                                                    config={"displayModeBar": True, "displaylogo": False},
                                                    style={"width": "100%"}
                                                ),
                                            ],
                                            pt="md"
                                        ),
                                        value="tendance",
                                    ),
                                ],
                                value="classement",
                            ),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="lg",
                        p="lg",
                    ),
                ],
                gap="lg",
            )
        ],
        fluid=True,
        p={"base": "md", "lg": "xl"},
        style={"background": COLORS['background'], "minHeight": "100vh"}
    )
)
