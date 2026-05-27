"""
POT Corner Plot
==============

Functions for creating corner plots for MCMC diagnostics.
"""

import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import corner
import base64
import io
import logging
import traceback
from typing import Dict


def create_corner_plot_gpd(stats: Dict, analysis_type: str) -> go.Figure:
    """Create a corner plot for GPD MCMC diagnostics with proper marginal histograms."""
    try:
        if 'mcmc_samples' not in stats or stats['mcmc_samples'] is None:
            return go.Figure().update_layout(
                title="Corner Plot (Not available)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )

        samples = stats['mcmc_samples']
        
        if analysis_type == 'bayesian_stationary':
            labels = ["Shape (ξ)", "Scale (σ)"]
            title_text = "Corner Plot: Stationary GPD Parameters"
        elif analysis_type == 'bayesian_non_stationary':
            labels = ["Shape (ξ)", "Scale₀ (σ₀)", "Scale₁ (σ₁)"]
            title_text = "Corner Plot: Non-Stationary GPD Parameters"
        else:
            return go.Figure()

        # Check if we have valid samples
        if samples.shape[0] == 0:
            return go.Figure().update_layout(title="No MCMC samples available")

        # Ensure clean matplotlib environment
        plt.style.use('default')
        plt.rcParams.update(plt.rcParamsDefault)
        
        # Set matplotlib parameters for high-quality output
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        
        # Create corner plot with maximum detail settings
        fig_mpl = corner.corner(
            samples, 
            labels=labels, 
            bins=30,  # More bins for smoother histograms
            truths=np.median(samples, axis=0),
            truth_color='red',
            # Enhanced label and title formatting
            label_kwargs=dict(fontsize=18, weight='bold'),
            title_kwargs=dict(fontsize=16, weight='bold'),
            # Histogram settings - matplotlib compatible only
            hist_kwargs=dict(
                alpha=0.8,  # Slight transparency for better visual
                color='#2E86AB',
                linewidth=1.0
                # Remove histtype - causes Line2D compatibility error
            ),
            # Scatter plot settings - optimized for visibility
            scatter_kwargs=dict(
                alpha=0.7,
                s=1.5,
                color='#2E86AB',
                rasterized=False  # Disable rasterization for vector quality
            ),
            # Enhanced contour settings
            contour_kwargs=dict(
                colors=['#2E86AB', '#A23B72', '#F18F01'],
                linewidths=[2.0, 1.5, 1.0],
                alpha=1.0
            ),
            contourf_kwargs=dict(
                colors=['#2E86AB'],
                alpha=0.4,
                levels=15  # More contour levels
            ),
            # Enable ALL plot components
            plot_density=True,
            plot_datapoints=True,
            plot_contours=True,
            fill_contours=True,
            # Enhanced quantile lines
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.4f',
            # Optimal smoothing
            smooth=1.5,
            smooth1d=1.5,
            # Figure configuration
            fig=None,
            # Additional parameters for completeness
            hist2d_kwargs=dict(alpha=0.8),
            range=None,  # Auto-range for all parameters
            max_n_ticks=5  # Clean axis ticks
        )
        
        # Optimal subplot spacing with maximum space for labels
        fig_mpl.subplots_adjust(
            left=0.18,
            bottom=0.30,  # Significantly increased for x-axis labels
            right=0.82,
            top=0.82,
            wspace=0.35,  # Maximum spacing for clarity
            hspace=0.35   # Maximum separation
        )
        
        # Set larger figure size for label visibility
        fig_mpl.set_size_inches(14, 12)  # Increased size for label space
        
        # Enhance ALL axes for maximum visibility
        axes = fig_mpl.get_axes()
        for i, ax in enumerate(axes):
            # Enhanced tick formatting with maximum visibility
            ax.tick_params(
                labelsize=14,  # Larger labels
                width=1.5, 
                length=6,
                direction='out',
                pad=12,  # Maximum padding for label visibility
                colors='black'
            )
            
            # Identify diagonal vs off-diagonal plots
            n_params = len(labels)
            row = i // n_params
            col = i % n_params
            
            if row == col:  # Diagonal histogram plots
                ax.grid(True, alpha=0.4, linewidth=1.0, linestyle='--', color='gray')
                ax.set_facecolor('white')
                # Enhance histogram appearance - ensure proper bars
                for patch in ax.patches:
                    patch.set_linewidth(1.0)
                    patch.set_edgecolor('black')
                    patch.set_alpha(0.8)  # Ensure bars are visible
            else:  # Off-diagonal contour plots
                ax.grid(True, alpha=0.2, linewidth=0.8, linestyle=':', color='lightgray')
                ax.set_facecolor('#fafafa')
            
            # Enhance all spines
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
                spine.set_color('black')
                spine.set_alpha(1.0)
            
            # Ensure labels are visible
            ax.xaxis.label.set_fontweight('bold')
            ax.yaxis.label.set_fontweight('bold')

        # Convert to high-quality PNG
        buf = io.BytesIO()
        fig_mpl.savefig(
            buf, 
            format='png', 
            dpi=150,  # High DPI optimized for container size
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.3,
            transparent=False,
            orientation='portrait',
            metadata={'Title': title_text, 'Creator': 'POT Dashboard'}
        )
        buf.seek(0)
        plt.close(fig_mpl)
        
        # Encode image
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Create Plotly figure with enhanced layout
        fig_go = go.Figure()
        
        # Add the image with better sizing
        fig_go.add_layout_image(
            dict(
                source=f'data:image/png;base64,{encoded_image}',
                xref="paper", yref="paper",
                x=0, y=1,
                sizex=1, sizey=1,
                xanchor="left", yanchor="top",
                sizing="contain",
                layer="below"
            )
        )
        
        fig_go.update_layout(
            title=dict(
                text=f"<b>{title_text}</b><br><i>Diagonal: Marginal Histograms | Off-diagonal: Parameter Correlations</i>",
                x=0.5,
                y=0.98,  # Position title higher to avoid overlap
                font=dict(size=20, weight='bold', color='#2c3e50')
            ),
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                visible=False,
                range=[0, 1]
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                visible=False,
                range=[0, 1]
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=900,  # Increased height
            width=1300,  # Increased width
            margin=dict(l=80, r=350, t=140, b=80),  # Increased margins for label space
            dragmode=False,
            showlegend=False
        )
        
        # Enhanced interpretation annotations
        summary_text = []
        diagnostic_text = []
        for i, label in enumerate(labels):
            param_samples = samples[:, i]
            median_val = np.median(param_samples)
            ci_lower = np.percentile(param_samples, 16)
            ci_upper = np.percentile(param_samples, 84)
            summary_text.append(f"{label}: {median_val:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Enhanced diagnostics
            if "Shape" in label:
                if -0.5 <= median_val <= 0.5:
                    diagnostic_text.append(f"[OK] {label}: Within typical range")
                elif median_val > 0.5:
                    diagnostic_text.append(f"[WARNING] {label}: Heavy-tailed distribution")
                else:
                    diagnostic_text.append(f"[WARNING] {label}: Light-tailed distribution")
            elif "Scale" in label:
                if median_val > 0:
                    diagnostic_text.append(f"[OK] {label}: Positive (valid)")
                else:
                    diagnostic_text.append(f"[ERROR] {label}: Must be positive")

        # Enhanced interpretation guide with visual cues
        fig_go.add_annotation(
            text=(
                "<b>Current Analysis Diagnostics:</b><br><br>" + "<br>".join(diagnostic_text)
            ),
            xref="paper", yref="paper",
            x=1.02, y=1.0,
            showarrow=False,
            font=dict(size=10, family="Arial"),
            bgcolor="rgba(248, 250, 252, 0.98)",
            bordercolor="#34495e",
            borderwidth=2,
            align="left",
            xanchor="left",
            yanchor="top"
        )
        
        # Enhanced parameter summary
        fig_go.add_annotation(
            text="<b>Parameter Summary (68% CIs):</b><br><br>" + "<br>".join(summary_text),
            xref="paper", yref="paper",
            x=1.02, y=0.45,
            showarrow=False,
            font=dict(size=11, family="monospace"),
            bgcolor="rgba(248, 249, 250, 0.95)",
            bordercolor="gray",
            borderwidth=1.5,
            align="left",
            xanchor="left",
            yanchor="top"
        )
        
        return fig_go

    except Exception as e:
        logging.error(f"Error creating GPD corner plot: {str(e)}")
        traceback.print_exc()
        return go.Figure().update_layout(
            title=f"Error creating corner plot: {e}",
            annotations=[
                dict(
                    text=f"Error: {str(e)}<br>Check logs for details",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color='red'),
                    align="center"
                )
            ]
        )

