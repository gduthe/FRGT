import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker
import torch
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def set_to_sci_format(ax_axis, num_ticks=6, font_size=10, offset_x=3.5, offset_y=1):
    """Set scientific notation formatting for axis with specified number of ticks."""
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    formatter.set_useOffset(True)
    formatter.orderOfMagnitude = 3  # Force the offset text to appear
    ax_axis.set_major_formatter(formatter)

    ax_axis.set_major_locator(ticker.MaxNLocator(num_ticks))
    ax_axis.get_offset_text().set_horizontalalignment('right')
    ax_axis.get_offset_text().set_verticalalignment('center')
    ax_axis.get_offset_text().set_x(offset_x)
    ax_axis.get_offset_text().set_y(offset_y)
    ax_axis.get_offset_text().set_fontsize(font_size)

def get_domain_bounds(graph_objects_list):
    """Get global domain bounds across all graphs."""
    xmin, xmax = float('inf'), float('-inf')
    ymin, ymax = float('inf'), float('-inf')
    
    for row in graph_objects_list:
        for graph in row:
            unique_points = torch.unique(graph.triangles)
            xmin = min(xmin, graph.triangle_points[0, unique_points].min().item())
            xmax = max(xmax, graph.triangle_points[0, unique_points].max().item())
            ymin = min(ymin, graph.triangle_points[1, unique_points].min().item())
            ymax = max(ymax, graph.triangle_points[1, unique_points].max().item())
    
    return xmin, xmax, ymin, ymax

def get_data_and_range(graph, node_type, channel):
    """Extract data for a given node type and channel."""
    nodes_mask = graph.node_type == node_type
    
    if isinstance(channel, tuple):
        # For velocity magnitude
        velocity_components = graph.y[nodes_mask][:, channel[0]:channel[1]+1]
        data = torch.norm(velocity_components, p=2, dim=1)
    else:
        # For single channel quantities
        data = graph.y[nodes_mask][:, channel]
        
    return data

def get_global_diff_bounds(baselines, predictions, plot_types):
    """
    Compute global difference bounds for each variable type across all rows.
    Returns a dictionary with difference bounds for each variable type.
    """
    diff_bounds = {'p': None, 'u_x': None, 'u_y': None, 'u_mag': None}
    
    for baseline, prediction, plot_type in zip(baselines, predictions, plot_types):
        if plot_type == 'p':
            channel = 0
        elif plot_type == 'u_x':
            channel = 1
        elif plot_type == 'u_y':
            channel = 2
        else:  # velocity_mag
            channel = (1, 2)
            
        baseline_data = get_data_and_range(baseline, 0, channel)
        prediction_data = get_data_and_range(prediction, 0, channel)
        diff_data = prediction_data - baseline_data
        
        diff_max = max(abs(diff_data.min()), abs(diff_data.max()))
        
        if diff_bounds[plot_type] is None or diff_max > diff_bounds[plot_type]:
            diff_bounds[plot_type] = diff_max
            
    return diff_bounds

def plot_single_model_comparison(baselines, predictions, plot_types='p', xlimits=None, ylimits=None, 
                         fig_size=(15, 4), dpi=100, show_triangles=True, show_colorbar=True,
                         num_ticks=6, cbar_font_size=12, title_font_size=12,
                         baseline_title="Baseline", prediction_title="Prediction",
                         add_farfield_info=False, colormaps=None):
    """
    Plot comparisons between baseline and prediction with difference for multiple samples.
    Difference plots for the same variable type will share the same colormap bounds.
    
    Parameters:
        baselines: List of baseline graph objects
        predictions: List of prediction graph objects
        plot_types: Either a single string or list of strings, each being one of:
                   'p' (pressure), 'u_x' (x-velocity), 'u_y' (y-velocity), 'u_mag' (velocity magnitude)
        xlimits: Optional tuple of (xmin, xmax) for plot bounds
        ylimits: Optional tuple of (ymin, ymax) for plot bounds
        fig_size: Base tuple of (width, height) for single row (will be scaled by number of rows)
        dpi: Resolution of the figure
        show_triangles: Whether to show mesh triangles
        show_colorbar: Whether to show colorbars
        num_ticks: Number of ticks on colorbars
        cbar_font_size: Font size for colorbar labels
        title_font_size: Font size for plot titles
        baseline_title: Title for baseline plots
        prediction_title: Title for prediction plots
        add_farfield_info: Add farfield information to the left of the first plot in each row
        colormaps: Optional list of colormaps for each row (defaults to 'viridis')
    """
    num_rows = len(baselines)
    assert len(predictions) == num_rows, "Number of baselines and predictions must match"
    
    # Handle plot types
    if isinstance(plot_types, str):
        plot_types = [plot_types] * num_rows
    assert len(plot_types) == num_rows, "Number of plot types must match number of rows"
    
    # Handle colormaps
    if colormaps is None:
        colormaps = ['viridis'] * num_rows
    elif isinstance(colormaps, str):
        colormaps = [colormaps] * num_rows
    assert len(colormaps) == num_rows, "Number of colormaps must match number of rows"
    
    # Convert to CPU if needed
    baselines = [b.cpu() for b in baselines]
    predictions = [p.cpu() for p in predictions]
    
    # Get global difference bounds for each variable type
    diff_bounds = get_global_diff_bounds(baselines, predictions, plot_types)
    
    # Adjust figure size for number of rows
    adjusted_height = fig_size[1] * num_rows
    fig = plt.figure(figsize=(fig_size[0], adjusted_height), dpi=dpi)
    
    # Create outer grid for rows
    outer_gs = GridSpec(num_rows, 1, figure=fig, height_ratios=[1]*num_rows, hspace=0.08)
    
    # Get global domain bounds if not provided
    if xlimits is not None:
        xmin, xmax = xlimits
    else:
        xmin = min(min(b.triangle_points[0].min() for b in baselines),
                  min(p.triangle_points[0].min() for p in predictions))
        xmax = max(max(b.triangle_points[0].max() for b in baselines),
                  max(p.triangle_points[0].max() for p in predictions))
    
    if ylimits is not None:
        ymin, ymax = ylimits
    else:
        ymin = min(min(b.triangle_points[1].min() for b in baselines),
                  min(p.triangle_points[1].min() for p in predictions))
        ymax = max(max(b.triangle_points[1].max() for b in baselines),
                  max(p.triangle_points[1].max() for p in predictions))
    
    # Plot each row
    for row, (baseline, prediction, plot_type, cmap) in enumerate(zip(baselines, predictions, plot_types, colormaps)):
        # Set up data type specific parameters for this row
        if plot_type == 'p':
            channel = 0
            cbar_label = 'p [Pa]'
            diff_label = 'Δp [Pa]'
        elif plot_type == 'u_x':
            channel = 1
            cbar_label = '$u_x$ [m/s]'
            diff_label = '$Δu_x$ [m/s]'
        elif plot_type == 'u_y':
            channel = 2
            cbar_label = '$u_y$ [m/s]'
            diff_label = '$Δu_y$ [m/s]'
        else:  # velocity_mag
            channel = (1,2)
            cbar_label = r'$\left\| \vec{u} \right\|$ [m/s]'
            diff_label = r'$Δ\left\| \vec{u} \right\|$ [m/s]'
        
        # Get row-specific value ranges
        baseline_data = get_data_and_range(baseline, 0, channel)
        prediction_data = get_data_and_range(prediction, 0, channel)
        diff_data = prediction_data - baseline_data
        
        vmin = min(baseline_data.min(), prediction_data.min())
        vmax = max(baseline_data.max(), prediction_data.max())
        
        # Use global difference bounds for this variable type
        diff_bounds['u_mag'] = 37
        diff_max = diff_bounds[plot_type]
        diff_vmin, diff_vmax = -diff_max, diff_max
        
        # Create row-specific grid
        width_ratios = [1, 1, 1] + ([0.05] if show_colorbar else [])
        row_gs = GridSpecFromSubplotSpec(1, len(width_ratios), 
                                       subplot_spec=outer_gs[row],
                                       width_ratios=width_ratios,
                                       wspace=0.05)
        
        # Create axes for this row
        axs = [fig.add_subplot(row_gs[0, i]) for i in range(3)]
        if show_colorbar:
            cax = fig.add_subplot(row_gs[0, -1])
        
        # Create triangulations
        baseline_tri = tri.Triangulation(x=baseline.triangle_points[0],
                                       y=baseline.triangle_points[1],
                                       triangles=baseline.triangles)
        prediction_tri = tri.Triangulation(x=prediction.triangle_points[0],
                                         y=prediction.triangle_points[1],
                                         triangles=prediction.triangles)
        
        # Plot baseline
        tpc1 = axs[0].tripcolor(baseline_tri, baseline_data, shading='flat', 
                               vmin=vmin, vmax=vmax, cmap=cmap)
        if show_triangles:
            axs[0].triplot(baseline_tri, color='black', alpha=0.2, lw=0.1)
        
        # Plot prediction
        tpc2 = axs[1].tripcolor(prediction_tri, prediction_data, shading='flat',
                               vmin=vmin, vmax=vmax, cmap=cmap)
        if show_triangles:
            axs[1].triplot(prediction_tri, color='black', alpha=0.2, lw=0.1)
        
        # Plot difference (always using RdBu_r for differences)
        tpc3 = axs[2].tripcolor(prediction_tri, diff_data, shading='flat',
                               vmin=diff_vmin, vmax=diff_vmax, cmap='RdBu_r')
        if show_triangles:
            axs[2].triplot(prediction_tri, color='black', alpha=0.2, lw=0.1)
        
        # Set titles only for first row
        titles = [baseline_title, prediction_title, "Difference"]
        for ax, title in zip(axs, titles):
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(title, fontsize=title_font_size)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add row label if required 
        Umag = baseline.globals_y[0,0].item()
        aoa = baseline.globals_y[0,1].item()
        Ti = baseline.globals_y[0,2].item()
        Re = Umag / 1.511e-05
        
        # Create row label with global info and plot type
        coeff = Re/10**int(np.log10(Re))
        exp = int(np.log10(Re))
        row_label = fr"Re=${coeff:.1f}\times10^{exp}$" + f"\nAoA={aoa:.1f}°"
        
        # Add text to the left of the first plot in the row
        if add_farfield_info:
            first_ax = axs[0]
            first_ax.text(-0.1, 0.5, row_label,
                transform=first_ax.transAxes, 
                verticalalignment='center',
                horizontalalignment='center',
                rotation=90,
                fontsize=cbar_font_size)
        
        # Add colorbars if requested
        if show_colorbar:
            # Colorbar for the difference plot
            cb1 = fig.colorbar(tpc3, cax=cax, orientation='vertical')
            cb1.ax.set_ylabel(diff_label, fontsize=cbar_font_size, rotation=270, labelpad=15)
            cb1.ax.tick_params(labelsize=cbar_font_size-2)
            
            if plot_type == 'p':
                set_to_sci_format(cb1.ax.yaxis, num_ticks, cbar_font_size-2, 4.5, 0)
            else:
                cb1.ax.yaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
    
    plt.tight_layout()
    return fig