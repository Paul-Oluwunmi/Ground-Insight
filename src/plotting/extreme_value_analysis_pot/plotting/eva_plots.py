"""
POT EVA Plots
=============

Functions for creating EVA (Extreme Value Analysis) plots.
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import traceback

from ..constants import SEASON_COLORS, COLORS
from ..statistical_calculations import calculate_return_periods


def _threshold_scalar(threshold):
    """Return a scalar for threshold (for adding to return levels)."""
    if threshold is None:
        return 0.0
    if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        return float(threshold)
    if hasattr(threshold, 'median'):
        return float(threshold.median())
    try:
        return float(np.median(threshold))
    except Exception:
        return 0.0


def update_eva_plots(exceedances, data_type, peaks_data=None, threshold=None, column_to_analyse=None):
    try:
        # Ensure we have enough data points for analysis
        if len(exceedances) < 2:
            raise ValueError("Insufficient data points for analysis. At least 2 peaks are required.")

        unit = 'm' if data_type == 'gwl' else 'mm'
        
        # Create main figure with subplots
        main_fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Seasonal Analysis',
                'Probability Density',
                'Return Level Plot',
                'Exceedance Probability',
                'Q-Q Plot',
                'P-P Plot'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
            column_widths=[0.5, 0.5],
            row_heights=[0.33, 0.33, 0.33]
        )

        location_name = column_to_analyse if column_to_analyse else 'Rainfall'
        title_text = f"Extreme Value Analysis - {location_name}"
        
        # Add EVA section title
        main_fig.add_annotation(
            text="Extreme Value Analysis (EVA) Results",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.15,
            showarrow=False,
            font=dict(
                size=24,
                color='rgb(44, 62, 80)',
                family='Arial, sans-serif',
                weight='bold'
            ),
            align='center'
        )

        main_fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                y=0.98,
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=28,
                    color='rgb(44, 62, 80)',
                    family='Arial, sans-serif',
                    weight='bold'
                ),
                pad=dict(b=20)
            ),
            margin=dict(t=120, l=80, r=80, b=80),  
            plot_bgcolor='white',
            paper_bgcolor='white',
            template='plotly_white',
            width=1600, 
            height=1600,  
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.3)',
                borderwidth=1
            )
        )

        # Update all subplots to be square
        for i in range(1, 4):  # 3 rows
            for j in range(1, 3):  # 2 columns
                main_fig.update_xaxes(
                    row=i, col=j,
                    showgrid=True,
                    gridwidth=0.5,
                    gridcolor='rgba(128, 128, 128, 0.5)',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='rgba(0, 0, 0, 0.5)',
                    showline=False,
                    linewidth=1,
                    linecolor='black',
                    mirror=True,  
                    scaleanchor=f"y{(i-1)*2 + j}",  
                    scaleratio=1  
                )
                main_fig.update_yaxes(
                    row=i, col=j,
                    showgrid=True,
                    gridwidth=0.5,
                    gridcolor='rgba(128, 128, 128, 0.5)',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='rgba(0, 0, 0, 0.5)',
                    showline=False,
                    linewidth=1,
                    linecolor='black',
                    mirror=True  
                )

        ret_periods = {}
        ret_periods['gpd_params'] = genpareto.fit(exceedances)
        
        # Calculate for Return Level Plot (row=1, col=1)
        theoretical_periods = np.logspace(np.log10(1), np.log10(100), 100)
        exceedance_probs = 1 / theoretical_periods
        theoretical_levels = genpareto.ppf(1 - exceedance_probs, *ret_periods['gpd_params'])
        ret_periods['theoretical_periods'] = theoretical_periods
        ret_periods['theoretical_levels'] = theoretical_levels
        
        # Add Return Level Plot
        main_fig.add_trace(
            go.Scattergl(
                x=theoretical_periods,
                y=theoretical_levels,
                mode='lines',
                name='Return Level',
                line=dict(color='red'),
                showlegend=True,
                hovertemplate=(
                    "Return Period: %{x:.1f} years<br>" +
                    f"{data_type.upper()}: %{{y:.2f}} {unit}<br>" +
                    "<extra></extra>"
                )
            ),
            row=2, col=1
        )

        # Add seasonal exceedance probability points
        if peaks_data is not None:
            for season in SEASON_COLORS:
                season_mask = peaks_data['Season'] == season
                if any(season_mask):
                    season_values = exceedances[season_mask]
                    season_dates = peaks_data[season_mask]['DateTime']
                    
                    # Calculate empirical exceedance probabilities
                    n = len(exceedances)
                    ranks = stats.rankdata(-season_values, method='average')
                    season_probs = ranks / (n + 1) * 100
                    
                    formatted_dates = [date.strftime('%d %b %Y') for date in season_dates]
                    
                    main_fig.add_trace(
                        go.Scattergl(
                            x=season_probs,
                            y=season_values,
                            mode='markers',
                            name=f'{season}',
                            marker=dict(color=SEASON_COLORS[season], size=8),
                            customdata=formatted_dates,
                            hovertemplate=(
                                f"{season}<br>" +
                                "Date: %{customdata}<br>" +
                                "Exceedance Probability: %{x:.1f}%<br>" +
                                f"{data_type.upper()}: %{{y:.2f}} {unit}<br>" +
                                "<extra></extra>"
                            )
                        ),
                        row=2, col=2
                    )

        return main_fig

    except Exception as e:
        logging.error(f"Error in EVA analysis: {str(e)}\nTraceback: {traceback.format_exc()}")
        raise


def create_full_eva_figure(exceedances, data_type, peaks_data, threshold, column_to_analyse, is_drought=False):
    """
    Build the full 6-panel EVA figure (Seasonal Analysis, Probability Density,
    Return Level Plot, Exceedance Probability, Q-Q Plot, P-P Plot) for stationary MLE.
    Matches the original dashboard logic so all panels are populated.
    """
    try:
        if len(exceedances) < 2:
            raise ValueError("Insufficient data points for analysis. At least 2 peaks are required.")
        if peaks_data is None or 'Season' not in peaks_data.columns:
            if peaks_data is not None and 'DateTime' in peaks_data.columns:
                peaks_data = peaks_data.copy()
                peaks_data['Season'] = peaks_data['DateTime'].dt.month.map({
                    12: 'Summer', 1: 'Summer', 2: 'Summer',
                    3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
                    6: 'Winter', 7: 'Winter', 8: 'Winter',
                    9: 'Spring', 10: 'Spring', 11: 'Spring'
                })
            else:
                raise ValueError("peaks_data with DateTime and Season required")

        unit = 'm' if data_type == 'gwl' else 'mm'
        th = _threshold_scalar(threshold)
        eva_results = calculate_return_periods(pd.Series(exceedances), data_type, is_drought)
        if eva_results is None:
            raise ValueError("Failed to calculate return periods")
        params = eva_results['gpd_params']

        main_fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Seasonal Analysis',
                'Probability Density',
                'Return Level Plot',
                'Exceedance Probability',
                'Q-Q Plot',
                'P-P Plot'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.12,
            column_widths=[0.5, 0.5],
            row_heights=[0.33, 0.33, 0.33]
        )

        location_name = column_to_analyse if column_to_analyse else 'Rainfall'
        x_label = 'Groundwater Level (m)' if data_type == 'gwl' else 'Rainfall (mm)'

        # ---- Row 1, Col 1: Seasonal Analysis ----
        y_col = 'mm_Rain' if data_type == 'rain' else peaks_data.columns[1]
        main_fig.add_trace(
            go.Scattergl(
                x=peaks_data['DateTime'],
                y=peaks_data[y_col],
                mode='lines+markers',
                name='Peaks',
                line=dict(color='gray', width=1),
                marker=dict(
                    color=[SEASON_COLORS[s] for s in peaks_data['Season']],
                    size=8,
                    symbol='circle'
                ),
                hovertemplate=(
                    "Season: %{customdata[0]}<br>Date: %{x|%d %b %Y}<br>"
                    f"{'GWL' if data_type == 'gwl' else 'Rainfall'}: %{{y:.2f}} {unit}<br>"
                    f"{'Deficit' if is_drought else 'Exceedance'}: %{{customdata[1]:.2f}} {unit}<br><extra></extra>"
                ),
                customdata=list(zip(peaks_data['Season'], exceedances))
            ),
            row=1, col=1
        )
        main_fig.update_xaxes(title_text="<b>Date</b>", row=1, col=1, gridcolor='white', gridwidth=0.5, showgrid=True)
        main_fig.update_yaxes(title_text=f"<b>{'Groundwater Level (m)' if data_type == 'gwl' else 'Rainfall (mm)'}</b>", row=1, col=1, gridcolor='white', gridwidth=0.5, showgrid=True)

        # ---- Return Level Plot (row 2, col 1) ----
        original_levels = eva_results['empirical_levels'] + th
        theoretical_levels = eva_results['theoretical_levels'] + th
        ci_lower = eva_results['ci_lower'] + th
        ci_upper = eva_results['ci_upper'] + th
        data_min = min(min(theoretical_levels), min(original_levels), min(ci_lower))
        data_max = max(max(theoretical_levels), max(original_levels), max(ci_upper))
        data_range = data_max - data_min
        y_min = round(data_min - 0.1 * data_range, 2)
        y_max = round(data_max + 0.1 * data_range, 2)
        data_span = y_max - y_min
        if data_type == 'gwl':
            dtick = 0.1 if data_span <= 0.5 else 0.2 if data_span <= 1.0 else 0.5 if data_span <= 2.0 else round(data_span / 6, 1)
        else:
            dtick = 1 if data_span <= 5 else 2 if data_span <= 10 else 5 if data_span <= 25 else 10 if data_span <= 50 else 20 if data_span <= 100 else 50 if data_span <= 250 else round(data_span / 6)

        rp_zones = [
            (1, 2, 'rgba(240, 248, 255, 0.3)', "Frequent (1-2 years)"),
            (2, 5, 'rgba(176, 224, 230, 0.3)', "Common (2-5 years)"),
            (5, 10, 'rgba(135, 206, 235, 0.3)', "Uncommon (5-10 years)"),
            (10, 25, 'rgba(70, 130, 180, 0.3)', "Rare (10-25 years)"),
            (25, 50, 'rgba(0, 0, 139, 0.3)', "Very Rare (25-50 years)"),
            (50, np.max(eva_results['theoretical_periods']), 'rgba(25, 25, 112, 0.3)', "Extreme (50+ years)")
        ]
        for (start, end, color, label) in rp_zones:
            main_fig.add_trace(
                go.Scattergl(x=[start, start, end, end], y=[y_min, y_max, y_max, y_min], fill='toself', fillcolor=color, line=dict(width=0), name=label, showlegend=True),
                row=2, col=1
            )
        for season in SEASON_COLORS:
            season_mask = peaks_data['Season'] == season
            if not any(season_mask):
                continue
            season_data = peaks_data[season_mask]
            original_values = season_data[y_col]
            sort_idx = np.argsort(original_values.values) if is_drought else np.argsort(-original_values.values)
            sorted_values = original_values.iloc[sort_idx]
            n = len(original_values)
            plotting_positions = np.arange(1, n + 1) / (n + 1)
            empirical_periods = 1 / plotting_positions if is_drought else 1 / (1 - plotting_positions)
            formatted_dates = season_data['DateTime'].iloc[sort_idx].dt.strftime('%d %b %Y')
            main_fig.add_trace(
                go.Scattergl(
                    x=empirical_periods,
                    y=sorted_values.values,
                    mode='markers',
                    name=f'{season} (n={sum(season_mask)})',
                    marker=dict(color=SEASON_COLORS[season], size=8, line=dict(color='white', width=1), symbol='circle'),
                    customdata=list(zip(formatted_dates, sorted_values.values)),
                    hovertemplate=f"<b>{season}</b><br>Date: %{{customdata[0]}}<br><b>Return Period:</b> %{{x:.1f}} years<br><b>{'GWL' if data_type == 'gwl' else 'Rainfall'}:</b> %{{y:.2f}} {unit}<br><extra></extra>"
                ),
                row=2, col=1
            )
        main_fig.add_trace(
            go.Scattergl(
                x=eva_results['theoretical_periods'],
                y=theoretical_levels,
                mode='lines',
                name='GPD Fit',
                line=dict(color='black', width=3, dash='solid'),
                showlegend=True
            ),
            row=2, col=1
        )
        for ci_values, ci_type in [(ci_lower, 'Lower'), (ci_upper, 'Upper')]:
            main_fig.add_trace(
                go.Scattergl(
                    x=eva_results['theoretical_periods'],
                    y=ci_values,
                    mode='lines',
                    line=dict(color='gray', dash='dash', width=1),
                    name='95% CI' if ci_type == 'Lower' else None,
                    showlegend=(ci_type == 'Lower'),
                    fill=None if ci_type == 'Lower' else 'tonexty',
                    fillcolor='rgba(128, 128, 128, 0.2)'
                ),
                row=2, col=1
            )
        main_fig.update_xaxes(title_text="Return Period (years)", type="log", range=[np.log10(1), np.log10(max(eva_results['theoretical_periods']))], row=2, col=1, gridcolor='lightgray', gridwidth=0.5, showgrid=True, ticktext=['1', '2', '5', '10', '20', '50', '100'], tickvals=[1, 2, 5, 10, 20, 50, 100])
        main_fig.update_yaxes(title_text=f"{'Groundwater Level (m)' if data_type == 'gwl' else 'Rainfall (mm)'}", range=[y_min, y_max], row=2, col=1, gridcolor='white', gridwidth=0.5, showgrid=True, tickformat='.2f', dtick=dtick)

        # ---- Q-Q Plot (row 3, col 1) ----
        actual_values = peaks_data[y_col]
        values_for_fit = -actual_values.values if is_drought else actual_values.values
        sorted_actual = np.sort(values_for_fit)
        theoretical_quantiles = genpareto.ppf(np.linspace(0.01, 0.99, len(actual_values)), *params)
        if is_drought:
            sorted_actual = -sorted_actual
            theoretical_quantiles = -theoretical_quantiles
        sorted_actual = sorted_actual + th
        theoretical_quantiles = theoretical_quantiles + th
        all_vals = np.concatenate([theoretical_quantiles, sorted_actual])
        min_val = np.min(all_vals) - 0.05 * (np.max(all_vals) - np.min(all_vals))
        max_val = np.max(all_vals) + 0.05 * (np.max(all_vals) - np.min(all_vals))
        qq_r2 = np.corrcoef(sorted_actual, theoretical_quantiles)[0, 1] ** 2
        qq_pearsonr, qq_pvalue = stats.pearsonr(sorted_actual, theoretical_quantiles)
        main_fig.add_trace(go.Scattergl(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(color='black', dash='dash', width=2), name='1:1', showlegend=False), row=3, col=1)
        for season in SEASON_COLORS:
            season_mask = peaks_data['Season'] == season
            if not any(season_mask):
                continue
            inds = np.where(season_mask.values)[0]
            season_vals = actual_values.iloc[inds]
            sort_idx = np.argsort(season_vals.values) if is_drought else np.argsort(-season_vals.values)
            season_vals_sorted = season_vals.iloc[sort_idx].values
            season_theo = theoretical_quantiles[inds][sort_idx]
            formatted_dates = peaks_data.loc[season_mask, 'DateTime'].iloc[sort_idx].dt.strftime('%d %b %Y')
            main_fig.add_trace(
                go.Scattergl(x=season_theo, y=season_vals_sorted, mode='markers', name=season, marker=dict(color=SEASON_COLORS[season], size=8, line=dict(color='white', width=1)), showlegend=False, customdata=list(zip(formatted_dates)), hovertemplate=f"{season}<br>Date: %{{customdata[0]}}<br>Theoretical: %{{x:.2f}} {unit}<br>Observed: %{{y:.2f}} {unit}<br><extra></extra>"),
                row=3, col=1
            )
        main_fig.add_annotation(text=f"R² = {qq_r2:.3f}<br>r = {qq_pearsonr:.3f}<br>p = {qq_pvalue:.3f}", xref="paper", yref="paper", x=0.02, y=0.20, showarrow=False, font=dict(size=10, weight='bold'), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1, align="left")
        main_fig.update_xaxes(title_text=f"Theoretical GPD Quantiles ({unit})", range=[min_val, max_val], row=3, col=1, gridcolor='white', gridwidth=0.5, showgrid=True)
        main_fig.update_yaxes(title_text=f"Sample Quantiles ({unit})", range=[min_val, max_val], row=3, col=1, gridcolor='white', gridwidth=0.5, showgrid=True)

        # ---- P-P Plot (row 3, col 2) ----
        if is_drought:
            empirical_probs = np.arange(1, len(actual_values) + 1) / (len(actual_values) + 1)
            values_for_cdf = -(actual_values.values - th)
            theoretical_probs = genpareto.cdf(values_for_cdf, *params)
        else:
            empirical_probs = 1 - (np.arange(1, len(actual_values) + 1) / (len(actual_values) + 1))
            theoretical_probs = 1 - genpareto.cdf(actual_values.values - th, *params)
        main_fig.add_trace(go.Scattergl(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', dash='dash', width=2), showlegend=False), row=3, col=2)
        for season in SEASON_COLORS:
            season_mask = peaks_data['Season'] == season
            if not any(season_mask):
                continue
            season_vals = actual_values[season_mask]
            n_season = len(season_vals)
            if is_drought:
                season_empirical = np.arange(1, n_season + 1) / (n_season + 1)
                season_theoretical = genpareto.cdf(-(season_vals.values - th), *params)
            else:
                season_empirical = 1 - (np.arange(1, n_season + 1) / (n_season + 1))
                season_theoretical = 1 - genpareto.cdf(season_vals.values - th, *params)
            formatted_dates = peaks_data[season_mask]['DateTime'].dt.strftime('%d %b %Y')
            main_fig.add_trace(
                go.Scattergl(x=season_empirical, y=season_theoretical, mode='markers', name=season, marker=dict(color=SEASON_COLORS[season], size=8, line=dict(color='white', width=1)), showlegend=False, customdata=list(zip(formatted_dates, season_vals)), hovertemplate=f"{season}<br>Date: %{{customdata[0]}}<br>Prob: %{{x:.3f}} / %{{y:.3f}}<br>Val: %{{customdata[1]:.2f}} {unit}<br><extra></extra>"),
                row=3, col=2
            )
        pp_r2 = np.corrcoef(empirical_probs, theoretical_probs)[0, 1] ** 2
        pp_pearsonr, pp_pvalue = stats.pearsonr(empirical_probs, theoretical_probs)
        main_fig.add_annotation(text=f"R² = {pp_r2:.3f}<br>r = {pp_pearsonr:.3f}<br>p = {pp_pvalue:.3f}", xref="paper", yref="paper", x=0.60, y=0.22, showarrow=False, font=dict(size=10, weight='bold'), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1)
        main_fig.update_xaxes(title_text=f"Empirical {'Non-exceedance' if is_drought else 'Exceedance'} Probability", range=[0, 1], row=3, col=2, gridcolor='white', gridwidth=0.5, showgrid=True)
        main_fig.update_yaxes(title_text="Theoretical Probability", range=[0, 1], row=3, col=2, gridcolor='white', gridwidth=0.5, showgrid=True)

        # ---- Probability Density (row 1, col 2) ----
        def _get_bins(data):
            if len(data) <= 1:
                return 10
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            if iqr == 0:
                return 10
            bw = 2 * iqr / (len(data) ** (1 / 3))
            dr = np.max(data) - np.min(data)
            if bw <= 0 or dr <= 0:
                return 10
            return max(10, min(int(np.ceil(dr / bw)), 50))

        actual_vals = actual_values.values
        n_bins = _get_bins(actual_vals)
        hist, bin_edges = np.histogram(actual_vals, bins=n_bins, density=True)
        for season in SEASON_COLORS:
            season_mask = peaks_data['Season'] == season
            if any(season_mask):
                main_fig.add_trace(
                    go.Histogram(
                        x=actual_values[season_mask].values,
                        name=f'{season} (n={sum(season_mask)})',
                        marker=dict(color=SEASON_COLORS[season], line=dict(color='white', width=0.5)),
                        opacity=0.7,
                        nbinsx=n_bins,
                        histnorm='probability density',
                        showlegend=True
                    ),
                    row=1, col=2
                )
        shape, loc, scale = params
        if len(actual_vals) > 0:
            x_range = np.linspace(np.min(actual_vals), np.max(actual_vals) + 0.2 * np.ptp(actual_vals), 500)
        else:
            x_range = np.linspace(0, 1, 500)
        pdf = genpareto.pdf(x_range - th, shape, loc, scale)
        hist_max = np.max(hist) if len(hist) > 0 else 1
        pdf_max = np.max(pdf) if np.max(pdf) > 0 else 1
        scaling = hist_max / pdf_max if pdf_max > 0 else 1
        pdf_scaled = pdf * scaling
        main_fig.add_trace(
            go.Scattergl(x=x_range, y=pdf_scaled, mode='lines', name=f'GPD PDF (ξ={shape:.3f})', line=dict(color='black', width=2.5, dash='solid'), showlegend=True),
            row=1, col=2
        )
        y_dens_max = max(hist_max, np.max(pdf_scaled)) * 1.1 if len(hist) or len(pdf_scaled) else 1
        main_fig.add_annotation(text=f"GPD: ξ={shape:.3f}, μ={loc:.3f}, σ={scale:.3f}<br>p-value={qq_pvalue:.3f}", xref="paper", yref="paper", x=1.06, y=0.99, xanchor="right", yanchor="top", showarrow=False, font=dict(size=10, weight='bold'), bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1, align="left")
        main_fig.update_xaxes(title_text=x_label, row=1, col=2, gridcolor='white', gridwidth=0.5, showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor='black')
        main_fig.update_yaxes(title_text="<b>Probability Density</b>", row=1, col=2, range=[0, y_dens_max], gridcolor='white', gridwidth=0.5, showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor='black')

        # ---- Exceedance Probability (row 2, col 2) ----
        exceedance_range = np.linspace(0, 1, 100)
        if is_drought:
            fitted_values = genpareto.ppf(exceedance_range, *params)
        else:
            fitted_values = genpareto.ppf(1 - exceedance_range, *params)
            fitted_values = np.where(np.isfinite(fitted_values), fitted_values, np.nan)
        if not np.all(np.isnan(fitted_values)):
            main_fig.add_trace(
                go.Scattergl(x=exceedance_range * 100, y=fitted_values, mode='lines', name='GPD Fit', line=dict(color='black', width=3, dash='solid'), showlegend=True),
                row=2, col=2
            )
        else:
            fitted_values = np.array([min(exceedances), max(exceedances)])
        n = len(exceedances)
        for season in SEASON_COLORS:
            season_mask = peaks_data['Season'] == season
            if not any(season_mask):
                continue
            season_vals = exceedances[season_mask]
            if is_drought:
                ranks = stats.rankdata(season_vals, method='average')
                season_probs = (n + 1 - ranks) / (n + 1) * 100
            else:
                ranks = stats.rankdata(-season_vals, method='average')
                season_probs = ranks / (n + 1) * 100
            formatted_dates = [d.strftime('%d %b %Y') for d in peaks_data[season_mask]['DateTime']]
            main_fig.add_trace(
                go.Scattergl(
                    x=season_probs,
                    y=season_vals.values,
                    mode='markers',
                    name=season,
                    marker=dict(color=SEASON_COLORS[season], size=8),
                    showlegend=False,
                    customdata=list(zip(formatted_dates, peaks_data.loc[season_mask, y_col])),
                    hovertemplate=f"{season}<br>Date: %{{customdata[0]}}<br>{'Non-exceedance' if is_drought else 'Exceedance'} Prob: %{{x:.1f}}%<br>{'Deficit' if is_drought else 'Exceedance'}: %{{y:.2f}} {unit}<br><extra></extra>"
                ),
                row=2, col=2
            )
        valid_fitted = fitted_values[np.isfinite(fitted_values)]
        y_min_exc = min(min(valid_fitted) if len(valid_fitted) > 0 else min(exceedances), min(exceedances))
        y_max_exc = max(max(valid_fitted) if len(valid_fitted) > 0 else max(exceedances), max(exceedances))
        y_range_exc = y_max_exc - y_min_exc
        main_fig.update_xaxes(title_text=f"<b>{'Non-exceedance' if is_drought else 'Exceedance'} Probability (%)</b>", range=[0, 100], row=2, col=2, gridcolor='white', gridwidth=0.5, showgrid=True)
        main_fig.update_yaxes(title_text=f"<b>{'Deficit' if is_drought else 'Exceedance'} ({unit})</b>", range=[y_min_exc - 0.1 * y_range_exc, y_max_exc + 0.1 * y_range_exc], row=2, col=2, gridcolor='white', gridwidth=0.5, showgrid=True)

        # ---- Layout ----
        main_fig.update_layout(
            title=dict(text=f"Extreme Value Analysis - {location_name}", x=0.5, y=0.99, font=dict(size=18, weight='bold')),
            height=1400,
            width=1400,
            showlegend=True,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            margin=dict(l=200, r=120, t=120, b=60),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.10, font=dict(size=12, weight='bold'))
        )
        _bold_font = dict(weight='bold')
        for k in range(6):
            r, c = (k // 2) + 1, (k % 2) + 1
            main_fig.update_xaxes(showgrid=True, gridcolor='lightgray', row=r, col=c, title_font=_bold_font, tickfont=_bold_font)
            main_fig.update_yaxes(showgrid=True, gridcolor='lightgray', row=r, col=c, title_font=_bold_font, tickfont=_bold_font)
        main_fig.update_yaxes(scaleanchor="x5", scaleratio=1, row=3, col=1)
        subplot_titles = ('Seasonal Analysis', 'Probability Density', 'Return Level Plot', 'Exceedance Probability', 'Q-Q Plot', 'P-P Plot')
        for ann in (getattr(main_fig.layout, 'annotations', None) or []):
            text = getattr(ann, 'text', None)
            if text in subplot_titles:
                main_fig.update_annotations(dict(font=dict(size=14, color='black', family='Arial', weight='bold')), selector=dict(text=text))

        return main_fig
    except Exception as e:
        logging.error(f"Error in create_full_eva_figure: {str(e)}\n{traceback.format_exc()}")
        raise

