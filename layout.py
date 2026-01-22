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
    COLORS,
    DELTA_COLORS,
    get_plotly_template,
)


EXCEL_PATH = "djingo_sat_aia_partitioned.xlsx"

df_init = get_data_bouchon()
MAX_YEAR, MAX_MONTH = latest_available_period(df_init)
DEFAULT_MONTH_VALUE = f"{MAX_YEAR}-{MAX_MONTH:02d}-01"
MIN_YEAR, MIN_MONTH = oldest_available_period(df_init)
MIN_MONTH_VALUE = f"{MIN_YEAR}-{MIN_MONTH:02d}-01"


def create_metric_card(title, value_id, subtitle_id=None, icon=None, sparkline_id=None, tooltip=None, subtitle_tooltip=None):
    """
    Create a KPI card with neutral background, right-aligned value, and optional sparkline.
    Following brand guidelines:
    - Background: Surface (neutral)
    - Title: Text muted
    - Value: Text primary
    - Spacing: 16px internal padding, 12px between title/value/delta
    - Number alignment: Right-align numeric values
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

    # Title with optional tooltip (info icon)
    title_element = dmc.Group(
        [
            dmc.Text(title, size="xs", c="dimmed", fw=500),
            dmc.Tooltip(
                label=tooltip,
                multiline=True,
                w=280,
                withArrow=True,
                position="top",
                children=DashIconify(
                    icon="material-symbols:info-outline",
                    width=14,
                    color=LIGHT_NEUTRALS['text_muted'],
                    style={"cursor": "help"}
                ),
            ) if tooltip else None,
        ],
        gap=4,
        align="center",
        mb=4,
    )

    card_content = [
        dmc.Group(
            [
                dmc.ThemeIcon(
                    DashIconify(icon=icon or "material-symbols:monitoring", width=20),
                    size="lg",
                    radius="md",
                    variant="light",
                    color="orange",  # Brand primary accent
                ),
                dmc.Box(flex=1),
            ],
            justify="space-between",
            mb=SPACING['card_internal_gap'],
        ),
        title_element,
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
        # Subtitle row with optional tooltip
        subtitle_label = dmc.Group(
            [
                dmc.Text("Sans récla:", size="xs", c="dimmed"),
                dmc.Tooltip(
                    label=subtitle_tooltip or "Valeur calculée en excluant les conversations liées à des réclamations.",
                    multiline=True,
                    w=250,
                    withArrow=True,
                    position="top",
                    children=DashIconify(
                        icon="material-symbols:info-outline",
                        width=12,
                        color=LIGHT_NEUTRALS['text_muted'],
                        style={"cursor": "help"}
                    ),
                ),
            ],
            gap=2,
            align="center",
        )
        card_content.append(
            dmc.Group(
                [
                    subtitle_label,
                    dmc.Text(
                        id=subtitle_id,
                        size="xs",
                        fw=500,
                        children="--",
                        style={"textAlign": "right"}
                    ),
                ],
                gap=4,
                mt=SPACING['card_internal_gap'],
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


# Définition des tooltips pour les métriques clés
METRIC_TOOLTIPS = {
    "dsat": (
        "DSAT (Delta Satisfaction) mesure l'écart entre les avis positifs et négatifs. "
        "• 0 = équilibre (autant de satisfaits que d'insatisfaits)\n"
        "• Positif (+) = plus de clients satisfaits\n"
        "• Négatif (-) = plus de clients insatisfaits"
    ),
    "vol_brut": (
        "Nombre total de conversations clients pour la période sélectionnée. "
        "Inclut toutes les interactions, qu'elles aient été notées ou non."
    ),
    "vol_note": (
        "Nombre de conversations ayant reçu une note de satisfaction de la part du client. "
        "Plus ce volume est élevé, plus l'indicateur DSAT est représentatif."
    ),
    "sans_recla": (
        "Valeur calculée en excluant les conversations liées à des réclamations. "
        "Permet d'isoler la satisfaction 'normale' du poids des réclamations."
    ),
}


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
                                    dmc.Group(
                                        [
                                            dmc.Title("Analyse DSAT", order=2, mb=4),
                                            dmc.Tooltip(
                                                label=(
                                                    "DSAT = Delta Satisfaction\n"
                                                    "Mesure la différence entre avis positifs et négatifs.\n\n"
                                                    "• DSAT > 0 : plus de clients satisfaits\n"
                                                    "• DSAT = 0 : équilibre\n"
                                                    "• DSAT < 0 : plus de clients insatisfaits"
                                                ),
                                                multiline=True,
                                                w=280,
                                                withArrow=True,
                                                position="right",
                                                children=DashIconify(
                                                    icon="material-symbols:help-outline",
                                                    width=20,
                                                    color=LIGHT_NEUTRALS['text_muted'],
                                                    style={"cursor": "help"}
                                                ),
                                            ),
                                        ],
                                        gap=8,
                                        align="center",
                                    ),
                                    dmc.Text("Tableau de bord de la satisfaction client par conversation", size="sm", c="dimmed"),
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
                                            dmc.Group(
                                                [
                                                    dmc.Text("Évolution DSAT et volumes sur l'année ...", size="lg", fw=600, id="main-graph-title"),
                                                    dmc.Tooltip(
                                                        label=(
                                                            "Ce graphique montre l'évolution mensuelle de la satisfaction :\n"
                                                            "• Ligne verte (Hors récla) = satisfaction sans les réclamations\n"
                                                            "• Ligne rose (Récla) = satisfaction globale\n"
                                                            "• Barres grises = volume total de conversations\n\n"
                                                            "L'écart entre les deux courbes montre l'impact des réclamations."
                                                        ),
                                                        multiline=True,
                                                        w=320,
                                                        withArrow=True,
                                                        position="bottom",
                                                        children=DashIconify(
                                                            icon="material-symbols:help-outline",
                                                            width=18,
                                                            color=LIGHT_NEUTRALS['text_muted'],
                                                            style={"cursor": "help"}
                                                        ),
                                                    ),
                                                ],
                                                gap=8,
                                                align="center",
                                            ),
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

                    # KPI Cards Row - Using brand neutral surface background
                    dmc.Grid(
                        [
                            dmc.GridCol(
                                create_metric_card(
                                    "DSAT Global",
                                    "dsat-main-value",
                                    "dsat-sans-value",
                                    icon="material-symbols:sentiment-dissatisfied",
                                    sparkline_id="dsat-spark",
                                    tooltip=METRIC_TOOLTIPS["dsat"],
                                    subtitle_tooltip=METRIC_TOOLTIPS["sans_recla"],
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
                                    tooltip=METRIC_TOOLTIPS["vol_brut"],
                                    subtitle_tooltip=METRIC_TOOLTIPS["sans_recla"],
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
                                    tooltip=METRIC_TOOLTIPS["vol_note"],
                                    subtitle_tooltip=METRIC_TOOLTIPS["sans_recla"],
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
                                        dmc.Group(
                                            [
                                                dmc.Text("Évolution vs M-1", size="sm", c="dimmed", fw=500),
                                                dmc.Tooltip(
                                                    label=(
                                                        "Comparaison avec le mois précédent (M-1). "
                                                        "• Valeur positive (+) = amélioration de la satisfaction\n"
                                                        "• Valeur négative (-) = dégradation de la satisfaction\n"
                                                        "• Un écart > 0.5 est considéré comme significatif"
                                                    ),
                                                    multiline=True,
                                                    w=280,
                                                    withArrow=True,
                                                    position="top",
                                                    children=DashIconify(
                                                        icon="material-symbols:info-outline",
                                                        width=14,
                                                        color=LIGHT_NEUTRALS['text_muted'],
                                                        style={"cursor": "help"}
                                                    ),
                                                ),
                                            ],
                                            gap=4,
                                            align="center",
                                            mb=4,
                                        ),
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
                                                dmc.Tooltip(
                                                    label=(
                                                        "Pilier : regroupe les données par grandes catégories métier (Facturation, Technique, etc.).\n"
                                                        "Intention : analyse plus fine par type de demande client détecté."
                                                    ),
                                                    multiline=True,
                                                    w=300,
                                                    withArrow=True,
                                                    position="bottom",
                                                    children=dmc.SegmentedControl(
                                                        id="grpby-select",
                                                        value="pilier",
                                                        data=[
                                                            {"label": "Par Pilier", "value": "pilier"},
                                                            {"label": "Par Intention", "value": "intention"},
                                                        ],
                                                        size="lg",
                                                    ),
                                                ),
                                                dmc.Group(
                                                    id="intention-controls",
                                                    children=[
                                                        dmc.Tooltip(
                                                            label="Nombre de catégories à afficher dans les graphiques.",
                                                            withArrow=True,
                                                            position="bottom",
                                                            children=dmc.Select(
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
                                                        ),
                                                        dmc.Tooltip(
                                                            label=(
                                                                "Fiabilité statistique : seules les catégories avec au moins 50 verbatims sont affichées. "
                                                                "Désactiver ce filtre peut montrer des résultats peu représentatifs."
                                                            ),
                                                            multiline=True,
                                                            w=280,
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
                                            dmc.Tooltip(
                                                label="Visualisez le classement des catégories par niveau de DSAT, avec et sans les réclamations.",
                                                multiline=True,
                                                w=250,
                                                withArrow=True,
                                                position="bottom",
                                                children=dmc.TabsTab(
                                                    dmc.Flex([
                                                        DashIconify(icon="material-symbols:bar-chart", width=18),
                                                        dmc.Text("Classement", size="sm", lh=10)
                                                    ], gap=8, align="center"),
                                                    value="classement",
                                                ),
                                            ),
                                            dmc.Tooltip(
                                                label="Matrice croisant DSAT et volume pour identifier les zones prioritaires d'action.",
                                                multiline=True,
                                                w=250,
                                                withArrow=True,
                                                position="bottom",
                                                children=dmc.TabsTab(
                                                    dmc.Flex([
                                                        DashIconify(icon="material-symbols:scatter-plot", width=18),
                                                        dmc.Text("Priorité", size="sm", lh=10)
                                                    ], gap=8, align="center"),
                                                    value="quadrants"
                                                ),
                                            ),
                                            dmc.Tooltip(
                                                label="Évolution temporelle des métriques sur plusieurs mois sous forme de heatmap.",
                                                multiline=True,
                                                w=250,
                                                withArrow=True,
                                                position="bottom",
                                                children=dmc.TabsTab(
                                                    dmc.Flex([
                                                        DashIconify(icon="material-symbols:calendar-view-month", width=18),
                                                        dmc.Text("Tendance", size="sm", lh=10)
                                                    ], gap=8, align="center"),
                                                    value="tendance"
                                                ),
                                            ),
                                            dmc.Tooltip(
                                                label="Tableau de données détaillé avec répartition des volumes par catégorie.",
                                                multiline=True,
                                                w=250,
                                                withArrow=True,
                                                position="bottom",
                                                children=dmc.TabsTab(
                                                    dmc.Flex([
                                                        DashIconify(icon="material-symbols:account-tree", width=18),
                                                        dmc.Text("Tableur", size="sm", lh=10)
                                                        ], gap=8, align="center",),
                                                    value="composition"
                                                ),
                                            ),
                                        ],
                                        grow=False,
                                    ),

                                    dmc.TabsPanel(
                                        dmc.Box(
                                            [
                                                dmc.Group(
                                                    [
                                                        dmc.Text(
                                                            "Classement des catégories par DSAT",
                                                            size="md",
                                                            c="dimmed",
                                                        ),
                                                        dmc.Tooltip(
                                                            label=(
                                                                "Ce graphique compare le DSAT avec réclamations (rose) "
                                                                "et sans réclamations (vert). L'écart entre les deux points "
                                                                "montre l'impact des réclamations sur la satisfaction. "
                                                                "Plus le segment est long, plus l'impact est fort."
                                                            ),
                                                            multiline=True,
                                                            w=300,
                                                            withArrow=True,
                                                            position="right",
                                                            children=DashIconify(
                                                                icon="material-symbols:help-outline",
                                                                width=18,
                                                                color=LIGHT_NEUTRALS['text_muted'],
                                                                style={"cursor": "help"}
                                                            ),
                                                        ),
                                                    ],
                                                    gap=8,
                                                    align="center",
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
                                                        dmc.Text("Données détaillées par catégorie", size="md", c="dimmed"),
                                                        dmc.Tooltip(
                                                            label=(
                                                                "Tableau interactif avec toutes les métriques par catégorie. "
                                                                "Cliquez sur les en-têtes pour trier. Le camembert montre la répartition des volumes."
                                                            ),
                                                            multiline=True,
                                                            w=280,
                                                            withArrow=True,
                                                            position="right",
                                                            children=DashIconify(
                                                                icon="material-symbols:help-outline",
                                                                width=18,
                                                                color=LIGHT_NEUTRALS['text_muted'],
                                                                style={"cursor": "help"}
                                                            ),
                                                        ),
                                                    ],
                                                    gap=8,
                                                    align="center",
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
                                                dmc.Group(
                                                    [
                                                        dmc.Text(
                                                            "Matrice de priorité DSAT × Volume",
                                                            size="md",
                                                            c="dimmed",
                                                        ),
                                                        dmc.Tooltip(
                                                            label=(
                                                                "Comment lire ce graphique :\n"
                                                                "• Axe horizontal = DSAT (satisfaction)\n"
                                                                "• Axe vertical = Volume de conversations\n"
                                                                "• Taille des bulles = Volume noté\n\n"
                                                                "Zone prioritaire : en haut à gauche (fort volume, faible DSAT) = maximum d'impact potentiel."
                                                            ),
                                                            multiline=True,
                                                            w=320,
                                                            withArrow=True,
                                                            position="right",
                                                            children=DashIconify(
                                                                icon="material-symbols:help-outline",
                                                                width=18,
                                                                color=LIGHT_NEUTRALS['text_muted'],
                                                                style={"cursor": "help"}
                                                            ),
                                                        ),
                                                    ],
                                                    gap=8,
                                                    align="center",
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
                                                        dmc.Group(
                                                            [
                                                                dmc.Text(
                                                                    "Évolution temporelle par catégorie",
                                                                    size="md",
                                                                    c="dimmed"
                                                                ),
                                                                dmc.Tooltip(
                                                                    label=(
                                                                        "Heatmap montrant l'évolution des métriques sur les derniers mois. "
                                                                        "Les couleurs vont du rose (mauvais) au vert (bon). "
                                                                        "Permet d'identifier les tendances et les catégories en amélioration ou dégradation."
                                                                    ),
                                                                    multiline=True,
                                                                    w=300,
                                                                    withArrow=True,
                                                                    position="right",
                                                                    children=DashIconify(
                                                                        icon="material-symbols:help-outline",
                                                                        width=18,
                                                                        color=LIGHT_NEUTRALS['text_muted'],
                                                                        style={"cursor": "help"}
                                                                    ),
                                                                ),
                                                            ],
                                                            gap=8,
                                                            align="center",
                                                        ),
                                                        dmc.Tooltip(
                                                            label=(
                                                                "• DSAT sans récla : satisfaction hors réclamations\n"
                                                                "• Δ (écart) : différence entre les deux DSAT\n"
                                                                "• DSAT total : satisfaction globale incluant réclamations"
                                                            ),
                                                            multiline=True,
                                                            w=280,
                                                            withArrow=True,
                                                            position="bottom",
                                                            children=dmc.SegmentedControl(
                                                                id="heatmap-metric",
                                                                value="dsat_total",
                                                                data=[
                                                                    {"label": "DSAT sans récla", "value": "dsat_sans"},
                                                                    {"label": "Δ (écart)", "value": "delta"},
                                                                    {"label": "DSAT total", "value": "dsat_total"},
                                                                ],
                                                                size="xs",
                                                            ),
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
        size=1280,  # Container max width per brand guidelines
        p={"base": SPACING['page_padding_mobile'], "lg": SPACING['page_padding_desktop']},
        style={"background": LIGHT_NEUTRALS['background'], "minHeight": "100vh"}
    )
)
