import time
from functools import lru_cache

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .theme import (
    BRAND_COLORS,
    SERIES_COLORS,
    DELTA_COLORS,
    LIGHT_NEUTRALS,
    DARK_NEUTRALS,
    get_chart_colors,
    get_heatmap_colorscale,
)

#region Config, chargement et mise en cache des données
query_partitioned_dataset = lambda x: x  # TODO: à remplacer par la vraie fonction

# Use brand colors for charts - mapped to semantic meaning
COLORS = {
    "primary": BRAND_COLORS["primary"],      # Orange #FF9A5B - brand accent
    "success": BRAND_COLORS["positive"],     # Teal #7FD1C3 - hors récla
    "warning": BRAND_COLORS["warning"],      # Gold #F3C969
    "danger": BRAND_COLORS["negative"],      # Raspberry #E07A8E - récla
    "teal": BRAND_COLORS["positive"],        # Teal #7FD1C3
    "purple": BRAND_COLORS["secondary"],     # Lavender #B9A7F8
    "orange": BRAND_COLORS["primary"],       # Orange #FF9A5B
    "gray": LIGHT_NEUTRALS["text_muted"],    # #5B677A
}
def cache_ttl(clear_time: int):
    """
    Décorateur pour invalider lru_cache toutes les clear_time secondes (TTL)
    :param clear_time: TTL en secondes
    """
    def inner(func):
        def wrapper(*args, **kwargs):
            if hasattr(func, 'next_clear'):
                # Si on a dépassé le délai, on clear le cache
                if time.time() > func.next_clear:
                    func.cache_clear()
                    func.next_clear = time.time() + clear_time
                else:
                    print("no read")
            else: # On ajoute l'attribut à la fonction pour y avoir accès directement au prochain call
                func.next_clear = time.time() + clear_time

            return func(*args, **kwargs)
        return wrapper
    return inner

# Commentez cette partie pour désactiver le cache (utile pour débugguer)
# ==============================================
@cache_ttl(86_400) # 86400 s = 1 jour
@lru_cache(1) # cache
# ==============================================
def get_data_bouchon() -> pd.DataFrame:
    """Charge le DataFrame local (bouchon). Utilise un cache pour éviter de recharger à chaque reload."""
    print("read")
    return pd.read_excel("djingo_sat_aia_partitioned.xlsx")

_DATA_CACHE = None
_DATA_VERSION = None

def load_data():
    """
    Charge les données depuis Dataiku avec un cache mémoire invalidé automatiquement
    dès qu'une nouvelle période est détectée.
    """
    global _DATA_CACHE, _DATA_VERSION

    # Récupère la dernière partition (période) depuis dataiku
    year_max, month_max = query_dataset_period_max("djingo_sat_aia_partitioned", date_col="dt")
    version = f"{year_max:04d}-{month_max:02d}"

    # On compare simplement si on à déjà cette période sinon on recharge le dataset
    if _DATA_CACHE is None or _DATA_VERSION != version:
        _DATA_CACHE = query_partitioned_dataset(
            dataset_name="djingo_sat_aia_partitioned",
            date_start="2025-01-01",
            date_end="2026-01-01",
            date_col="dt",
        )
        _DATA_VERSION = version

    return _DATA_CACHE
#endregion

#region Gestion des périodes
def parse_month_value(value):
    """Convertit une valeur MonthPickerInput 'YYYY-MM' en tuple (year:int, month:int)."""
    if value is None:
        raise ValueError("Month value cannot be None")
    dt = pd.to_datetime(value).to_pydatetime()
    return dt.year, dt.month


def previous_period(year: int, month: int):
    """Retourne la période précédente (année, mois)."""
    if month == 1:
        return year - 1, 12
    return year, month - 1


def latest_available_period(df: pd.DataFrame):
    """
    Détecte la dernière période disponible dans le DataFrame.
    Retourne: year, month
    """
    p = df["year"] * 100 + df["month"]
    idx = p.idxmax()
    return int(df.loc[idx, "year"]), int(df.loc[idx, "month"])

def oldest_available_period(df: pd.DataFrame):
    """
    Détecte la plus ancienne période disponible dans le DataFrame.
    Retourne: year, month
    """
    p = df["year"] * 100 + df["month"]
    idx = p.idxmin()
    return int(df.loc[idx, "year"]), int(df.loc[idx, "month"])


def month_label(year: int, month: int) -> str:
    """Formate une période en chaîne 'YYYY-MM'."""
    return f"{year}-{month:02d}"


def get_current_period(df: pd.DataFrame, month_value):
    """
    Retourne (year, month) en fonction de la valeur sélectionnée
    ou de la dernière période dispo si None.
    """
    if month_value is None:
        return latest_available_period(df)
    return parse_month_value(month_value)
#endregion

#region Sélection et aggrégation des données
def select_row(
    df: pd.DataFrame,
    year: int,
    month: int,
    level: str,
    category: str,
    type_filtrage: str = "min 50",
) -> pd.Series:
    """
    Sélectionne la ligne unique (year, month, niveau, categorie, type_filtrage).

    Hypothèse : le couple (year, month, level, category, type_filtrage) est unique.
    """
    mask = (
        (df["year"] == year)
        & (df["month"] == month)
        & (df["niveau"] == level)
        & (df["categorie"] == category)
    )
    if "type_filtrage" in df.columns:
        mask &= df["type_filtrage"] == type_filtrage

    subset = df.loc[mask]

    if subset.empty:
        raise KeyError(
            f"Aucune ligne trouvée pour {year=}, {month=}, niveau={level}, "
            f"categorie={category}, type_filtrage={type_filtrage}"
        )

    return subset.iloc[0]


def select_timeseries(df: pd.DataFrame, niveau: str, categorie: str) -> pd.DataFrame:
    """
    Historique agrégé pour un couple (niveau, catégorie).

    Retourne les colonnes year, month, delta_sat, vol_brut + period formatée.
    """
    mask = (df["niveau"] == niveau) & (df["categorie"] == categorie)
    cols = ["year", "month", "delta_sat", "vol_brut"]
    ts = df.loc[mask, cols].copy()
    ts["period"] = ts.apply(
        lambda r: month_label(int(r["year"]), int(r["month"])), axis=1
    )
    return ts.sort_values(["year", "month"])


def compute_volumes_for_period(df: pd.DataFrame, year: int, month: int, subset="all"):
    """
    Calcule (vol_brut, vol_note) pour une période et un sous-ensemble.

    subset:
      - "all"        -> niveau="globale", categorie="globale"
      - "sans_recla" -> niveau="globale_distinct_faire_recla", categorie="Globale"
    """
    if subset == "all":
        row = select_row(df, year, month, "globale", "globale")
    elif subset == "sans_recla":
        row = select_row(df, year, month, "globale_distinct_faire_recla", "Globale")
    else:
        raise ValueError(f"subset inconnu: {subset}")

    vol_brut = float(row["vol_brut"])
    taux_rep = float(row["taux_reponse_au_sondage"])
    vol_note = vol_brut * taux_rep
    return vol_brut, vol_note


def base_label(cat: str) -> str:
    """
    Retourne le label 'de base' en retirant le suffixe '_réclamation' s'il existe.

    Exemple :
        'Facturation_réclamation' -> 'Facturation'
    """
    if isinstance(cat, str) and cat.endswith("_réclamation"):
        return cat[: -len("_réclamation")]
    return cat


def aggregate_kpis(
    df: pd.DataFrame,
    year: int,
    month: int,
    groupby: str = "pilier",
    type_filtrage: str = "min 50"
) -> pd.DataFrame:
    """
    Agrège les KPI par pilier / intention pour un mois donné.

    Retourne un DataFrame avec colonnes :
    ['label', 'dsat_total', 'dsat_sans', 'delta', 'vol_brut', 'vol_note']
    """
    groupy_recla = groupby + "_distinct_faire_recla"

    mask_period = (df["year"] == year) & (df["month"] == month)
    mask_tf = (df["type_filtrage"] == type_filtrage) if type_filtrage else np.ones(len(df), dtype=bool)
    # si filtrage est none on ne filtre pas

    cols = ["categorie", "delta_sat", "vol_brut", "vol_note", "taux_reponse_au_sondage", "type_filtrage"]

    # Vue agrégée "totale"
    d_total = df.loc[mask_period & mask_tf & (df["niveau"] == groupby), cols].copy()
    d_total = d_total.rename(
        columns={
            "categorie": "label",
            "delta_sat": "dsat_total",
            "vol_brut": "vol_brut_total",
            "vol_note": "vol_note_total",
            "taux_reponse_au_sondage": "taux_rep_total",
        }
    )

    # Cas trivial : si pas de données et groupby != pilier
    if d_total.empty and groupby != "pilier":
        return pd.DataFrame(
            columns=["label", "dsat_total", "dsat_sans", "delta", "vol_brut", "vol_note"]
        )

    # Cas pilier : gestion avec / sans réclamation
    if groupby == "pilier":
        if groupy_recla:
            d_split = df.loc[
                mask_period & mask_tf & (df["niveau"] == groupy_recla), cols
            ].copy()
        else:
            d_split = pd.DataFrame(columns=cols)

        if not d_split.empty:
            d_split["label_base"] = d_split["categorie"].apply(base_label)
            d_split["is_recla"] = d_split["categorie"].astype(str).str.endswith(
                "_réclamation"
            )

            d_sans = d_split.loc[~d_split["is_recla"]].copy()
            d_sans = d_sans.rename(
                columns={
                    "delta_sat": "dsat_sans",
                    "vol_brut": "vol_brut_sans",
                    "vol_note": "vol_note_sans",
                    "taux_reponse_au_sondage": "taux_rep_sans",
                }
            )
            d_sans["label"] = d_sans["label_base"]
            d_sans = d_sans[["label", "dsat_sans", "vol_brut_sans", "taux_rep_sans"]]
        else:
            d_sans = pd.DataFrame(
                columns=["label", "dsat_sans", "vol_brut_sans", "taux_rep_sans"]
            )

        if not d_total.empty:
            d_total = d_total.groupby("label", as_index=False).agg(
                {
                    "dsat_total": "mean",
                    "vol_brut_total": "sum",
                    "taux_rep_total": "mean",
                    "vol_note_total": "sum",
                }
            )

        has_split = not d_sans.empty
        merged = pd.merge(d_sans, d_total, on="label", how="outer").fillna(0.0)

        if not has_split:
            # Si pas de découpage sans/avec récla, on cale dsat_sans sur dsat_total
            merged["dsat_sans"] = merged.get("dsat_total", 0.0)

        merged["delta"] = merged["dsat_sans"] - merged["dsat_total"]
        merged["vol_brut"] = merged["vol_brut_total"]
        merged["vol_note"] = merged["vol_note_total"]

        out = merged[
            ["label", "dsat_total", "dsat_sans", "delta", "vol_brut", "vol_note"]
        ].sort_values("vol_brut", ascending=False)
        return out

    # Autres groupby : pas de split avec/sans récla, on recopie dsat_total
    if d_total.empty:
        return pd.DataFrame(
            columns=["label", "dsat_total", "dsat_sans", "delta", "vol_brut", "vol_note"]
        )

    d_total = d_total.groupby("label", as_index=False).agg(
        {
            "dsat_total": "mean",
            "vol_brut_total": "sum",
            "vol_note_total": "sum",
            "taux_rep_total": "mean",
        }
    )

    d_total["dsat_sans"] = d_total["dsat_total"]
    d_total["delta"] = 0.0
    d_total["vol_brut"] = d_total["vol_brut_total"]
    d_total["vol_note"] = d_total["vol_note_total"]

    out = d_total[
        ["label", "dsat_total", "dsat_sans", "delta", "vol_brut", "vol_note"]
    ].sort_values("vol_brut", ascending=False)
    return out
#endregion

#region Calcul des métriques (KPI cards)
def compute_metrics(df: pd.DataFrame, year: int, month: int) -> dict:
    """
    Calcule les principaux indicateurs globaux pour une période donnée.

    Retourne un dict avec toutes les valeurs nécessaires pour le bandeau de métriques.
    """
    # Avec réclamation
    row_all = select_row(df, year, month, "globale", "globale")
    vol_brut_all = float(row_all["vol_brut"])
    taux_rep_all = float(row_all["taux_reponse_au_sondage"])
    vol_note_all = vol_brut_all * taux_rep_all
    dsat_total = float(row_all["delta_sat"])

    # Sans réclamation
    row_sans = select_row(df, year, month, "globale_distinct_faire_recla", "Globale")
    dsat_sans = float(row_sans["delta_sat"])
    vol_brut_sans = float(row_sans["vol_brut"])
    taux_rep_sans = float(row_sans["taux_reponse_au_sondage"])
    vol_note_sans = vol_brut_sans * taux_rep_sans

    return {
        "vol_brut_all": vol_brut_all,
        "vol_note_all": vol_note_all,
        "dsat_total": dsat_total,
        "dsat_sans": dsat_sans,
        "vol_brut_sans": vol_brut_sans,
        "vol_note_sans": vol_note_sans,
    }


def build_insight_reclamations(metrics: dict) -> dict:
    """
    Construit le texte et le style de la carte d'insight sur l'impact des réclamations.

    metrics : dict retourné par compute_metrics.
    """
    dsat_total = metrics["dsat_total"]
    dsat_sans = metrics["dsat_sans"]
    vol_brut_all = metrics["vol_brut_all"]
    vol_brut_sans = metrics["vol_brut_sans"]

    impact_recla = dsat_sans - dsat_total
    pct_sans_recla = (vol_brut_sans / vol_brut_all * 100) if vol_brut_all > 0 else 0.0

    insight_text = (
        f"Les réclamations ont un impact de {impact_recla:.2f} points de DSAT. "
        f"Les réclamations représentent {100 - pct_sans_recla:.1f}% du volume "
        "mais dégradent fortement la satisfaction."
    )

    return {
        "insight_text": insight_text,
        "insight_color": "blue",
        "insight_title": "📊 Zone d'Attention",
    }

def build_delta_card(df: pd.DataFrame, current_year: int, current_month: int, dsat_total: float):
    """
    Calcule les informations d'évolution de DSAT vs le mois précédent
    pour alimenter la carte 'delta DSAT'.

    Brand colors for deltas:
    - Positive (improvement): Teal
    - Negative (degradation): Raspberry
    - Warning/Stable: Gold
    """
    try:
        prev_year, prev_month = previous_period(current_year, current_month)
        row_prev = select_row(df, prev_year, prev_month, "globale", "globale")
        dsat_prev = float(row_prev["delta_sat"])
        dsat_diff = dsat_total - dsat_prev

        sign = "+" if dsat_diff > 0 else ""
        delta_display = f"{sign}{dsat_diff:.2f}"

        if dsat_diff < -0.5:
            # Dégradation - use raspberry (negative)
            delta_color = "raspberry"
            icon_color = "raspberry"
            badge_text = "Dégradation"
            trend_text = (
                f"La DSAT a diminué de {abs(dsat_diff):.2f} points vs le mois précédent"
            )
        elif dsat_diff > 0.5:
            # Amélioration - use teal (positive)
            delta_color = "teal"
            icon_color = "teal"
            badge_text = "Amélioration"
            trend_text = (
                f"La DSAT a augmenté de {abs(dsat_diff):.2f} points vs le mois précédent"
            )
        else:
            # Stable - use gold (warning/neutral)
            delta_color = "gold"
            icon_color = "orange"
            badge_text = "Stable"
            trend_text = "La DSAT reste stable par rapport au mois précédent"

    except Exception:
        delta_display = "N/A"
        delta_color = "gray"
        icon_color = "gray"
        badge_text = "Pas de comparaison"
        trend_text = "Données du mois précédent non disponibles"

    return delta_display, delta_color, icon_color, badge_text, trend_text
#endregion

#region Graphiques communs (placeholders, erreurs)
def build_empty_spark_figure():
    """Figure vide standard pour les cas d'erreur pour les sparklines."""
    chart_colors = get_chart_colors("light")
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=50,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=chart_colors["plot_bgcolor"],
    )
    return fig


def build_error_placeholder_figure(message="Erreur de chargement des données"):
    """Figure placeholder standard pour les erreurs sur les graphs principaux."""
    chart_colors = get_chart_colors("light")
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        font=dict(size=14, color=chart_colors["text_muted"]),
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=chart_colors["plot_bgcolor"],
    )
    return fig
#endregion

#region Graphique principal (trend) et sparklines
def _configure_sparkline(fig, values, color=None, lock_zero=False):
    """Configure une sparkline compacte avec un design moderne utilisant les couleurs de la marque."""
    if color is None:
        color = BRAND_COLORS["primary"]

    vmin = float(min(values))
    vmax = float(max(values))
    if lock_zero and vmin >= 0:
        vmin = 0.0

    span = max(1e-9, vmax - vmin)
    pad = span * 0.08
    y0, y1 = vmin - pad, vmax + pad

    rgb = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    fig.update_traces(
        line=dict(color=color, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba{rgb + (0.1,)}",
    )

    chart_colors = get_chart_colors("light")

    fig.update_layout(
        height=50,
        margin=dict(b=0, t=0, l=0, r=0),
        xaxis=dict(visible=True, fixedrange=True),
        yaxis=dict(
            range=[y0, y1],
            fixedrange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
        ),
        plot_bgcolor=chart_colors["plot_bgcolor"],
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
        ),
        hovermode="x unified",
    )

    return fig


def build_metric_sparklines(df: pd.DataFrame, current_period=None, n_periods=12):
    """
    Construit les sparklines pour :
      - volume brut,
      - volume noté,
      - DSAT globale.
    """
    if current_period is None:
        y, m = latest_available_period(df)
    else:
        y, m = current_period

    labels, brut_vals, note_vals, sat_vals = [], [], [], []
    seen = 0
    cur_y, cur_m = y, m

    while seen < n_periods and cur_y is not None and cur_m is not None:
        try:
            row = select_row(df, cur_y, cur_m, "globale", "globale")
            vb = float(row["vol_brut"])
            vn = float(row["vol_note"])
            sat = float(row["delta_sat"])
        except Exception:
            break

        labels.append(month_label(cur_y, cur_m))
        brut_vals.append(vb)
        note_vals.append(vn)
        sat_vals.append(sat)

        seen += 1
        cur_y, cur_m = previous_period(cur_y, cur_m)

    labels = labels[::-1]
    brut_vals = brut_vals[::-1]
    note_vals = note_vals[::-1]
    sat_vals = sat_vals[::-1]

    if not labels:
        empty = build_empty_spark_figure()
        return empty, empty, empty

    fig_brut = go.Figure(
        go.Scatter(
            x=labels,
            y=brut_vals,
            mode="lines",
            showlegend=False,
            hovertemplate="%{x}<br>%{y:,.0f} conversations<extra></extra>",
        )
    )

    fig_note = go.Figure(
        go.Scatter(
            x=labels,
            y=note_vals,
            mode="lines",
            showlegend=False,
            hovertemplate="%{x}<br>%{y:,.0f} réponses<extra></extra>",
        )
    )

    fig_sat = go.Figure(
        go.Scatter(
            x=labels,
            y=sat_vals,
            mode="lines",
            showlegend=False,
            hovertemplate="%{x}<br>DSAT: %{y:,.2f}<extra></extra>",
        )
    )

    # Use brand colors: Primary orange for volume, Teal for positive metrics
    fig_brut = _configure_sparkline(fig_brut, brut_vals, color=BRAND_COLORS["primary"])
    fig_note = _configure_sparkline(fig_note, note_vals, color=BRAND_COLORS["positive"])
    fig_sat = _configure_sparkline(
        fig_sat, sat_vals, color=BRAND_COLORS["positive"], lock_zero=False
    )

    return fig_brut, fig_note, fig_sat


def build_main_graph():
    """
    Construit le graphique principal d'évolution DSAT avec / sans réclamation
    + volume brut en barres (axe secondaire).

    Brand colors applied:
    - DSAT sans récla (Hors récla): Teal #7FD1C3
    - DSAT avec récla (Récla): Raspberry #E07A8E
    - Volume bars: Primary orange with low opacity
    """
    df = get_data_bouchon()
    chart_colors = get_chart_colors("light")

    ts_no = select_timeseries(
        df, "globale_distinct_faire_recla", "Globale"
    ).rename(columns={"delta_sat": "dsat_no"})
    ts_re = select_timeseries(df, "globale", "globale").rename(
        columns={"delta_sat": "dsat_re"}
    )

    cols_re = ["year", "month", "period", "dsat_re"]
    if "vol_brut" in ts_re.columns:
        cols_re.append("vol_brut")

    merged = (
        pd.merge(
            ts_no[["year", "month", "period", "dsat_no"]],
            ts_re[cols_re],
            on=["year", "month", "period"],
            how="inner",
        )
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

    if merged.empty:
        return go.Figure()

    merged["delta"] = merged["dsat_no"] - merged["dsat_re"]

    n = len(merged)
    x_idx = np.arange(n, dtype=float)
    periods = merged["period"].tolist()

    fig = go.Figure()
    bar_width = 0.8

    # Volume brut en barres (fond) - using brand primary with low opacity
    if "vol_brut" in merged.columns:
        # Convert brand primary to rgba with low opacity
        primary_rgb = tuple(int(BRAND_COLORS["primary"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        fig.add_trace(
            go.Bar(
                x=x_idx,
                y=merged["vol_brut"],
                name="Volume brut",
                marker=dict(color=f"rgba({primary_rgb[0]}, {primary_rgb[1]}, {primary_rgb[2]}, 0.12)"),
                yaxis="y2",
                width=bar_width,
                hovertemplate="%{customdata}<br>Volume brut: %{y:,.0f}<extra></extra>",
                customdata=periods,
            )
        )

    # Zone entre les deux courbes DSAT - using brand colors for fill
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_idx, x_idx[::-1]]),
            y=np.concatenate([merged["dsat_no"], merged["dsat_re"][::-1]]),
            fill="toself",
            fillcolor="rgba(224, 122, 142, 0.15)",  # Raspberry with low opacity
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="Écart",
        )
    )

    # Courbe DSAT sans récla (Hors récla) - Teal
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=merged["dsat_no"],
            mode="lines+markers",
            name="Hors récla",
            line=dict(color=SERIES_COLORS["hors_recla"], width=3),
            marker=dict(
                size=8,
                color=SERIES_COLORS["hors_recla"],
                line=dict(width=2, color="white"),
            ),
            hovertemplate="%{customdata}<br>DSAT hors récla: %{y:.2f}<extra></extra>",
            customdata=periods,
        )
    )

    # Courbe DSAT avec récla (Récla) - Raspberry
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=merged["dsat_re"],
            mode="lines+markers",
            name="Récla",
            line=dict(color=SERIES_COLORS["recla"], width=3),
            marker=dict(
                size=8,
                color=SERIES_COLORS["recla"],
                line=dict(width=2, color="white"),
            ),
            hovertemplate=(
                "%{customdata[0]}<br>"
                "DSAT récla: %{y:.2f}<br>"
                "Écart: %{customdata[1]:.2f}<extra></extra>"
            ),
            customdata=np.column_stack([periods, merged["delta"]]),
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=40),
        height=320,
        xaxis=dict(
            type="linear",
            showgrid=False,
            tickmode="array",
            tickvals=x_idx,
            ticktext=periods,
            tickangle=-45,
            tickfont=dict(size=10, color=chart_colors["text_muted"]),
            range=[x_idx[0], x_idx[-1]],
        ),
        yaxis=dict(
            title="DSAT",
            title_font_size=12,
            title_font_color=chart_colors["text_muted"],
            showgrid=True,
            gridcolor=chart_colors["gridcolor"],
            zeroline=True,
            zerolinecolor=chart_colors["border"],
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            showticklabels=True,
            title="Volume brut",
            title_font_size=11,
            title_font_color=chart_colors["text_muted"],
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=chart_colors["plot_bgcolor"],
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=chart_colors["text_color"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=chart_colors["border"],
            borderwidth=1,
        ),
        hovermode="x unified",
        bargap=0,
    )

    return fig
#endregion

#region Graphiques des onglets (1, 2, 3)
def build_fig_dumbbell(df_agg: pd.DataFrame, top_n=12, sort_by="dsat_sans"):
    """
    Graphique 'dumbbell' comparant DSAT total vs DSAT sans réclamation.

    Brand colors:
    - Récla (total): Raspberry #E07A8E
    - Hors récla (sans): Teal #7FD1C3
    """
    if df_agg.empty:
        return build_error_placeholder_figure("Aucune donnée disponible")

    chart_colors = get_chart_colors("light")

    sort_by = sort_by if sort_by in df_agg.columns else "dsat_sans"
    d = df_agg.sort_values(sort_by, ascending=False).head(top_n).copy()
    d["ycat"] = pd.Categorical(d["label"], categories=list(d["label"])[::-1], ordered=True)
    y = d["ycat"]

    v = d["vol_note"].astype(float)
    if v.max() > 0:
        msize = 10 + 18 * (v - v.min()) / (v.max() - v.min() + 1e-9)
    else:
        msize = 12

    # Segments de connexion - using raspberry (negative) with opacity
    x_lines, y_lines = [], []
    for _, r in d.iterrows():
        x_lines += [r["dsat_total"], r["dsat_sans"], None]
        y_lines += [r["ycat"], r["ycat"], None]

    # Convert raspberry to rgba
    raspberry_rgb = tuple(int(BRAND_COLORS["negative"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_lines,
            y=y_lines,
            mode="lines",
            line=dict(color=f"rgba({raspberry_rgb[0]}, {raspberry_rgb[1]}, {raspberry_rgb[2]}, 0.4)", width=4),
            showlegend=False,
            hoverinfo="skip",
            name="Écart",
        )
    )

    # Récla (total) - Raspberry
    fig.add_trace(
        go.Scatter(
            x=d["dsat_total"],
            y=y,
            mode="markers",
            name="Récla",
            marker=dict(
                color=SERIES_COLORS["recla"],
                size=msize,
                line=dict(width=2, color="white"),
                symbol="circle",
            ),
            hovertemplate=(
                "%{y}<br>"
                "DSAT récla: %{x:.2f}<br>"
                "Écart (hors - récla): %{customdata[0]:.2f}<br>"
                "Vol. noté: %{customdata[1]:,.0f}<extra></extra>"
            ),
            customdata=d[["delta", "vol_note"]].to_numpy(),
        )
    )

    # Hors récla (sans) - Teal
    fig.add_trace(
        go.Scatter(
            x=d["dsat_sans"],
            y=y,
            mode="markers",
            name="Hors récla",
            marker=dict(
                color=SERIES_COLORS["hors_recla"],
                size=msize,
                line=dict(width=2, color="white"),
                symbol="circle",
            ),
            hovertemplate=(
                "%{y}<br>"
                "DSAT hors récla: %{x:.2f}<br>"
                "Écart (hors - récla): %{customdata[0]:.2f}<br>"
                "Vol. noté: %{customdata[1]:,.0f}<extra></extra>"
            ),
            customdata=d[["delta", "vol_note"]].to_numpy(),
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(300, top_n * 35),
        xaxis=dict(
            title="DSAT",
            zeroline=True,
            zerolinecolor=chart_colors["border"],
            showgrid=True,
            gridcolor=chart_colors["gridcolor"],
        ),
        yaxis=dict(title=None, showgrid=True, gridcolor=chart_colors["gridcolor"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=chart_colors["border"],
            borderwidth=1,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=chart_colors["plot_bgcolor"],
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=chart_colors["text_color"],
        ),
    )

    return fig

def build_fig_quadrants(df_agg: pd.DataFrame):
    """
    Scatter "quadrants" : DSAT sans récla vs volume brut.

    Brand colors:
    - Colorscale: Raspberry (low/bad) -> Gold (mid) -> Teal (high/good)
    - Quadrant fills use brand colors with low opacity
    """
    if df_agg.empty:
        return build_error_placeholder_figure("Aucune donnée disponible")

    chart_colors = get_chart_colors("light")

    d = df_agg.copy()
    x = d["dsat_sans"].astype(float)
    y = d["vol_brut"].astype(float)
    x_med = float(x.median()) if len(x) else 0.0
    y_med = float(y.median()) if len(y) else 0.0

    fig = go.Figure()

    # Convert brand colors to rgba for quadrant fills
    raspberry_rgb = tuple(int(BRAND_COLORS["negative"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    teal_rgb = tuple(int(BRAND_COLORS["positive"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    # Zones des quadrants
    fig.add_shape(
        type="rect",
        x0=x.min(),
        x1=x_med,
        y0=y_med,
        y1=y.max(),
        fillcolor=f"rgba({raspberry_rgb[0]}, {raspberry_rgb[1]}, {raspberry_rgb[2]}, 0.05)",
        line=dict(width=0),
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=x_med,
        x1=x.max(),
        y0=y_med,
        y1=y.max(),
        fillcolor=f"rgba({teal_rgb[0]}, {teal_rgb[1]}, {teal_rgb[2]}, 0.05)",
        line=dict(width=0),
        layer="below",
    )

    # Lignes médianes using border color
    fig.add_shape(
        type="line",
        x0=x_med,
        x1=x_med,
        y0=y.min(),
        y1=y.max(),
        line=dict(color=chart_colors["text_muted"], width=2, dash="dot"),
    )
    fig.add_shape(
        type="line",
        x0=x.min(),
        x1=x.max(),
        y0=y_med,
        y1=y_med,
        line=dict(color=chart_colors["text_muted"], width=2, dash="dot"),
    )

    # Points with brand colorscale
    fig.add_trace(
        go.Scatter(
            x=d["dsat_sans"],
            y=d["vol_brut"],
            mode="markers+text",
            marker=dict(
                size=d["vol_note"] / d["vol_note"].max() * 40 + 10,
                color=d["dsat_sans"],
                colorscale=[
                    [0, BRAND_COLORS["negative"]],     # Raspberry for low DSAT (bad)
                    [0.5, BRAND_COLORS["warning"]],    # Gold for mid
                    [1, BRAND_COLORS["positive"]],     # Teal for high DSAT (good)
                ],
                line=dict(width=2, color="white"),
                showscale=True,
                colorbar=dict(title="DSAT<br>hors récla", thickness=15, len=0.7),
            ),
            text=d["label"],
            textposition="top center",
            textfont=dict(size=9, color=chart_colors["text_color"]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "DSAT hors récla: %{x:.2f}<br>"
                "Volume: %{y:,.0f}<br>"
                "Vol. noté: %{customdata:,.0f}<extra></extra>"
            ),
            customdata=d["vol_note"],
            showlegend=False,
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=640,
        xaxis=dict(
            title="DSAT hors récla",
            showgrid=True,
            gridcolor=chart_colors["gridcolor"],
            zeroline=True,
            zerolinecolor=chart_colors["border"],
        ),
        yaxis=dict(
            title="Volume brut",
            showgrid=True,
            gridcolor=chart_colors["gridcolor"],
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=chart_colors["plot_bgcolor"],
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=chart_colors["text_color"],
        ),
    )
    return fig


def build_fig_heatmap(
    df: pd.DataFrame,
    year: int,
    month: int,
    groupby="pilier",
    metric="dsat_sans",
    top_n=10,
    n_periods=6,
    type_filtrage="min 50",
):
    """
    Heatmap d'évolution par label sur plusieurs périodes.

    Brand colors:
    - Colorscale: Raspberry (low/bad) -> Gold (mid) -> Teal (high/good)
    - Text labels adapt based on background luminance for readability
    """
    chart_colors = get_chart_colors("light")

    # Construction de la liste de périodes
    periods = []
    y, m = year, month
    for _ in range(n_periods):
        periods.append((y, m, month_label(y, m)))
        y, m = previous_period(y, m)
    periods = periods[::-1]

    frames = []
    for (yy, mm, per_lab) in periods:
        agg = aggregate_kpis(
            df, yy, mm, groupby=groupby, type_filtrage=type_filtrage
        )
        if agg.empty:
            continue

        cur = agg.copy()
        cur["period"] = per_lab
        if metric == "dsat_sans":
            cur["value"] = cur["dsat_sans"]
        elif metric == "delta":
            cur["value"] = cur["delta"]
        else:
            cur["value"] = cur["dsat_total"]

        frames.append(cur[["label", "period", "value", "vol_brut"]])

    if not frames:
        return build_error_placeholder_figure("Aucune donnée disponible")

    D = pd.concat(frames, ignore_index=True)

    # Sélection des labels dominants en volume
    top_labels = (
        D.groupby("label")["vol_brut"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    D = D[D["label"].isin(top_labels)]

    mat = D.pivot_table(
        index="label", columns="period", values="value", aggfunc="mean"
    ).fillna(0.0)
    ordered_cols = [p[2] for p in periods if p[2] in mat.columns]
    mat = mat.loc[top_labels, ordered_cols]

    metric_label = {
        "dsat_sans": "DSAT hors récla",
        "delta": "Δ (hors - récla)",
        "dsat_total": "DSAT récla",
    }.get(metric, metric)

    # Brand colorscale: Raspberry -> Gold -> Teal
    colorscale = get_heatmap_colorscale(metric)

    fig = go.Figure(
        go.Heatmap(
            z=mat.values,
            x=mat.columns.tolist(),
            y=mat.index.tolist(),
            text=np.round(mat.values, 2),
            texttemplate="%{text:.2f}",
            textfont=dict(color=chart_colors["text_primary"], size=11),
            colorscale=colorscale,
            zmid=0.0 if metric in ("dsat_sans", "delta") else None,
            hoverongaps=False,
            hovertemplate="%{y} · %{x}<br>%{z:.2f} pts<extra></extra>",
            colorbar=dict(title=metric_label, thickness=15, len=0.7),
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(300, top_n * 40),
        xaxis=dict(
            side="top",
            tickangle=-45,
            tickfont=dict(size=10, color=chart_colors["text_muted"])
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=10, color=chart_colors["text_muted"])
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=chart_colors["plot_bgcolor"],
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=chart_colors["text_color"],
        ),
        hovermode="x unified",
    )

    return fig
#endregion

#region AgGrid & camembert de composition (onglet 4)
def build_grid_columns(groupby: str):
    """Construit la définition des colonnes du grid de composition."""
    return [
        {"field": "label", "headerName": groupby.capitalize(), "pinned": "left"},
        {"field": "vol_brut", "headerName": "Volume brut", "type": "numericColumn"},
        {"field": "vol_note", "headerName": "Volume noté", "type": "numericColumn"},
        {"field": "dsat_total", "headerName": "DSAT total", "type": "numericColumn"},
        {"field": "dsat_sans", "headerName": "DSAT sans récla", "type": "numericColumn"},
    ]


def build_grid_style(visible: bool = True):
    """Style du grid, visible ou caché."""
    display = "block" if visible else "none"
    return {"width": "100%", "height": "420px", "display": display}


def build_pie_composition_figure(agg: pd.DataFrame, groupby: str):
    """
    Construit le camembert de répartition des volumes.

    Uses brand color sequence for consistency.
    """
    chart_colors = get_chart_colors("light")

    # Brand color sequence for pie slices
    brand_sequence = [
        BRAND_COLORS["primary"],      # Orange
        BRAND_COLORS["positive"],     # Teal
        BRAND_COLORS["negative"],     # Raspberry
        BRAND_COLORS["warning"],      # Gold
        BRAND_COLORS["secondary"],    # Lavender
        "#8B9AAF",                     # Muted blue-gray
        "#D4A574",                     # Muted tan
        "#9BC1BC",                     # Muted sage
        "#C9A7C7",                     # Muted mauve
        "#A8C686",                     # Muted lime
    ]

    fig = px.pie(
        agg,
        names="label",
        values="vol_brut",
        title=f"Répartition des volumes par {groupby}",
        hole=0.35,
        color_discrete_sequence=brand_sequence,
    )
    fig.update_traces(
        hovertemplate="%{label}<br>%{value:,.0f} cas (%{percent})<extra></extra>",
        textposition="inside",
        texttemplate="%{percent:.1%}",
        textfont=dict(color="white"),
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=chart_colors["text_color"],
        ),
    )
    return fig


def build_pie_style(visible: bool = True):
    """Style du camembert, visible ou caché."""
    return {"display": "block" if visible else "none"}
#endregion
