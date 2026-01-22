from dash import Input, Output
from satisfaction_djingo.function import *


# Fonction appelée par le main
def register_callbacks(app):
    #region Callbacks des métriques
    @app.callback(
        Output("metrics-store", "data"),
        Input("month-select", "value"),
    )
    def update_metrics(month_value):
        """Callback pour mettre à jour les valeurs aggrégées selon la période.
        Les valeurs sont dans un store qui servira d'input aux autres callbacks"""
        df = None
        try:
            df = get_data_bouchon()
            year, month = get_current_period(df, month_value)

            # Metrics cards
            metrics = compute_metrics(df, year, month)
            # Sparklines
            spark_brut, spark_note, spark_dsat = build_metric_sparklines(df, n_periods=12)
            # Delta card
            delta_display, delta_color, icon_color, badge_text, trend_text = build_delta_card(
                df, year, month, metrics["dsat_total"]
            )
            # Insight réclamations
            insight = build_insight_reclamations(metrics)

            return {
                "year": year,
                "month": month,
                "metrics": metrics,
                "sparklines": {
                    "vol_brut": spark_brut,
                    "vol_note": spark_note,
                    "dsat": spark_dsat,
                },
                "delta": {
                    "display": delta_display,
                    "color": delta_color,
                    "icon_color": icon_color,
                    "badge_text": badge_text,
                    "trend_text": trend_text,
                },
                "insight": insight,
            }

        except Exception as e:
            print(f"Error inupdate_metrics: {e}")
            # On renvoie une structure cohérente mais marquée comme erreur
            if df is None:
                print("Unable to load dataframe, it is undefined.")
            return {
                "error": True,
            }

    @app.callback(
        Output("vol-brut-value", "children"),
        Output("vol-note-value", "children"),
        Output("dsat-main-value", "children"),
        Output("dsat-sans-value", "children"),
        Output("vol-brut-sans-value", "children"),
        Output("vol-note-sans-value", "children"),
        Output("main-graph-title", "children"),
        Input("metrics-store", "data"),
    )
    def update_main_values(data):
        """
        Met à jour les valeurs numériques affichées dans le bandeau à partir du Store.
        """
        if not data or data.get("error"):
            return "--", "--", "--", "--", "--", "--", "Évolution DSAT et volumes sur l'année --"

        m = data["metrics"]
        return (
            f"{m['vol_brut_all']:,.0f}".replace(",", " "),
            f"{m['vol_note_all']:,.0f}".replace(",", " "),
            f"{m['dsat_total']:.2f}",
            f"{m['dsat_sans']:.2f}",
            f"{m['vol_brut_sans']:,.0f}".replace(",", " "),
            f"{m['vol_note_sans']:,.0f}".replace(",", " "),
            f"Évolution DSAT et volumes sur l'année {data['year']}"
        )

    @app.callback(
        Output("vol-brut-spark", "figure"),
        Output("vol-note-spark", "figure"),
        Output("dsat-spark", "figure"),
        Input("metrics-store", "data"),
    )
    def update_sparklines(data):
        """
        Met à jour les mini-graphes d'évolution (sparklines).
        """
        if not data or data.get("error"):
            empty = build_empty_spark_figure()
            return empty, empty, empty

        s = data["sparklines"]
        return s["vol_brut"], s["vol_note"], s["dsat"]

    @app.callback(
        Output("dsat-delta-value", "children"),
        Output("dsat-delta-badge", "children"),
        Output("dsat-delta-badge", "color"),
        Output("delta-icon", "color"),
        Output("delta-trend-text", "children"),
        Input("metrics-store", "data"),
    )
    def update_delta_card_view(data):
        """
        Met à jour la carte indiquant l'évolution de la DSAT vs M-1.
        """
        if not data or data.get("error"):
            return "N/A", "Erreur", "gray", "gray", "Erreur de chargement"

        d = data["delta"]
        return (
            d["display"],
            d["badge_text"],
            d["color"],
            d["icon_color"],
            d["trend_text"],
        )

    @app.callback(
        Output("insight-card", "children"),
        Output("insight-card", "color"),
        Output("insight-card", "title"),
        Input("metrics-store", "data"),
    )
    def update_insight_card(data):
        """
        Met à jour la carte d'insight sur l'impact des réclamations.
        """
        if not data or data.get("error"):
            return "Impossible de charger les données.", "red", "Erreur"

        i = data["insight"]
        return i["insight_text"], i["insight_color"], i["insight_title"]
    #endregion
    #region Callbacks des onglets
    @app.callback(
        Output("intention-controls", "style"),
        Input("grpby-select", "value"),
    )
    def toggle_intention_controls(groupby):
        base_style = {
            "gap": "1rem",
            "flexWrap": "wrap",
            "alignItems": "center",
            "display": "flex",
        }
        if groupby != "intention":
            base_style["display"] = "none"
        return base_style

    @app.callback(
        Output("fig-classement", "figure"),
        Output("composition-grid", "rowData"),
        Output("composition-grid", "columnDefs"),
        Output("composition-grid", "style"),
        Output("composition-grid-skeleton", "visible"),
        Output("composition-pie", "figure"),
        Output("composition-pie", "style"),
        Output("composition-pie-skeleton", "visible"),
        Output("fig-quadrants", "figure"),
        Output("fig-tendance", "figure"),
        Input("month-select", "value"),
        Input("grpby-select", "value"),
        Input("top-n-select", "value"),
        Input("heatmap-metric", "value"),
        Input("filter-min50-checkbox", "checked"),
    )
    def update_kpi_tabs(month_value, groupby, top_n_str, heat_metric, use_filtering):
        """
        Met à jour les 4 onglets de KPI en fonction de leur paramètres:
        - de la période sélectionnée,
        - du regroupement (groupby),
        - du top N,
        - de la métrique de la heatmap.
        """
        try:
            df = get_data_bouchon()
            year, month = get_current_period(df, month_value)
            top_n = int(top_n_str) if top_n_str else 10
            filtrage = "min 50" if (use_filtering and groupby == "intention") else None
            agg = aggregate_kpis(df, year, month, groupby=groupby, type_filtrage=filtrage)

            # Graphiques (dumbbell, quadrants, heatmap)
            fig_classement = build_fig_dumbbell(agg, top_n=top_n, sort_by="dsat_sans")
            fig_quadrants = build_fig_quadrants(agg.head(top_n))
            fig_tendance = build_fig_heatmap(
                df, year, month,
                groupby=groupby,
                metric=heat_metric,
                top_n=top_n,
                n_periods=6,
                type_filtrage=filtrage,
            )

            # Onglet 4, tableur + camembert
            grid_cols = build_grid_columns(groupby)
            grid_style = build_grid_style(visible=True)
            skeleton_visible = False

            pie_fig = build_pie_composition_figure(agg, groupby)
            pie_style = build_pie_style(visible=True)
            pie_skeleton_visible = False

            return (
                fig_classement,
                agg.to_dict("records"),
                grid_cols,
                grid_style,
                skeleton_visible,
                pie_fig,
                pie_style,
                pie_skeleton_visible,
                fig_quadrants,
                fig_tendance,
            )

        except Exception as e:
            print(f"Error in update_kpi_tabs: {e}")

            placeholder = build_error_placeholder_figure()
            return (
                placeholder,  # fig-classement
                [],  # grid rowData
                [],  # grid columnDefs
                build_grid_style(visible=False),  # grid style
                True,  # grid skeleton visible
                placeholder,  # pie figure
                build_pie_style(visible=False),  # pie style
                True,  # pie skeleton visible
                placeholder,  # fig-quadrants
                placeholder,  # fig-tendance
            )
    #endregion
