'''
This script generates plots to help visualize the simulated data
It was made with the help of LLMs
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors


PATH_FINAL_DATA = './raw_output/simdata_final_model.csv'
PATH_NOPI_DATA = './raw_output/simdata_noPI_model.csv'

PL_FLUX = 'vCdsA'
LP_FLUX = 'vLpxK'


# Unified sequence of all 27 target metabolites for clean sequential grid mapping
METABOLITES = [
    # 1. CXX acyl-ACPs (12 variables)
    ('y15', 'C4:0', 'skyblue'),
    ('y20', 'C6:0', 'skyblue'),
    ('y25', 'C8:0', 'skyblue'),
    ('y30', 'C10:0', 'skyblue'),
    ('y36', 'C12:0', 'skyblue'),
    ('y41', 'C12:1Z', 'mediumpurple'),
    ('y46', 'C14:0', 'skyblue'),
    ('y51', 'C14:1Z', 'mediumpurple'),
    ('y56', 'C16:0', 'skyblue'),
    ('y61', 'C16:1Z', 'mediumpurple'),
    ('y66', 'C18:0', 'skyblue'),
    ('y71', 'C18:1Z', 'mediumpurple'),
    
    # 2. CXX beta-hydroxy-ACPs (14 variables)
    ('y13', 'C4:0-OH', 'firebrick'),
    ('y18', 'C6:0-OH', 'firebrick'),
    ('y23', 'C8:0-OH', 'firebrick'),
    ('y28', 'C10:0-OH', 'firebrick'),
    ('y34', 'C12:0-OH', 'firebrick'),
    ('y39', 'C12:1-OH', 'orange'),
    ('y44', 'C14:0-OH', 'firebrick'),
    ('y49', 'C14:1-OH', 'orange'),
    ('y54', 'C16:0-OH', 'firebrick'),
    ('y59', 'C16:1-OH', 'orange'),
    ('y64', 'C18:0-OH', 'firebrick'),
    ('y69', 'C18:1-OH', 'orange'),
    ('y74', 'C20:0-OH', 'firebrick'),
    ('y79', 'C20:1-OH', 'orange'),
    
    # 3. Cis-Unsaturated Intermediate (1 variable)
    ('y32', 'C10:1Z', 'mediumpurple')
]


def remove_unevaluated_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies columns containing unevaluated Mathematica expressions
    and removes the entire column from the DataFrame, keeping only 
    purely numerical concentration and flux vectors.
    """
    initial_cols = df.columns.tolist()
    
    # 1. Force all columns to numeric. Non-convertible strings become NaN
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # 2. Drop COLUMNS (axis=1) that contain any NaN values
    df_clean = df_numeric.dropna(axis=1)
    
    # 3. Track which specific columns failed evaluation for your records
    kept_cols = df_clean.columns.tolist()
    dropped_cols = [col for col in initial_cols if col not in kept_cols]
    
    print(f"Cleaned DataFrame: Kept {len(kept_cols)}/{len(initial_cols)} columns.")
    if dropped_cols:
        print(f"--> Dropped non-numerical columns: {dropped_cols}\n")
        
    return df_clean


def log2_normalize_to_average(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Applies a log2 transformation to the specified column data after normalizing to the average.
    """
    # Using np.log2 to compute the base-2 logarithm element-wise
    return np.log2(df[col] / df[col].mean())


def plot_condition_variance_proof(df: pd.DataFrame, df_nopi: pd.DataFrame) -> None:
    # Initialize a 3x1 single-column layout for a clean vertical fit in a paper
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))
    
    # Pre-sort datasets by MalCoA to ensure continuous line tracing for subplots 1 & 2
    df_sort_mal = df.sort_values(by='MalCoA')
    df_nopi_sort_mal = df_nopi.sort_values(by='MalCoA')
    
    # --- Subplot 1: Mal-CoA vs. Phospholipid Flux ---
    # Control Data (Darkturquoise)
    axes[0].plot(
        df_sort_mal['MalCoA'], 
        df_sort_mal[PL_FLUX], 
        'o-', 
        color='darkturquoise', 
        linewidth=1.5, 
        markersize=4,
        alpha=0.85,
        label='PlsB Product Inhibition'
    )
    # noPI Data (Cyan)
    axes[0].plot(
        df_nopi_sort_mal['MalCoA'], 
        df_nopi_sort_mal[PL_FLUX], 
        'o-', 
        color='cyan', 
        linewidth=1.5, 
        markersize=4,
        alpha=0.85,
        label='No Product Inhibition'
    )
    axes[0].set_title(r'A. Phospholipid Flux vs. Mal-CoA', fontsize=11, fontweight='bold')
    axes[0].set_xlabel(r'Mal-CoA Conc. [$\mu$M]', fontsize=10)
    axes[0].set_ylabel(r'Phospholipid Flux [$\frac{\mu \mathrm{M}}{s}$]', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.3, color='gray')
    axes[0].set_xlim(0, 750)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc='lower right', fontsize=9, framealpha=0.7)
    
    # --- Subplot 2: Mal-CoA vs. Phosphatidic Acid ---
    # Control Data (Dark Orange)
    axes[1].plot(
        df_sort_mal['MalCoA'], 
        df_sort_mal['yP3'], 
        'o-', 
        color='darkorange', 
        linewidth=1.5, 
        markersize=4,
        alpha=0.85,
        label='PlsB Product Inhibition'
    )
    # noPI Data (Gold)
    axes[1].plot(
        df_nopi_sort_mal['MalCoA'], 
        df_nopi_sort_mal['yP3'], 
        'o-', 
        color='gold', 
        linewidth=1.5, 
        markersize=4,
        alpha=0.85,
        label='No Product Inhibition'
    )
    axes[1].set_title(r'B. Phosphatidic Acid vs. Mal-CoA', fontsize=11, fontweight='bold')
    axes[1].set_xlabel(r'Mal-CoA Conc. [$\mu$M]', fontsize=10)
    axes[1].set_ylabel(r'PA Conc. [$\mu$M]', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.3, color='gray')
    axes[1].set_xlim(0, 750)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc='lower right', fontsize=9, framealpha=0.7)
    
    # --- Subplot 3: Phospholipid Flux vs. Phosphatidic Acid ---
    # Sort independently by each dataset's respective flux column acting as the X-axis
    df_sort_flux = df.sort_values(by=PL_FLUX)
    df_nopi_sort_flux = df_nopi.sort_values(by=PL_FLUX)
    
    # Control Data (Mediumpurple)
    axes[2].plot(
        log2_normalize_to_average(df_sort_flux, PL_FLUX), 
        log2_normalize_to_average(df_sort_flux, 'yP3'), 
        'o-', 
        color='mediumpurple', 
        linewidth=1.5, 
        markersize=4,
        alpha=0.85
    )
    
    axes[2].axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    axes[2].axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    axes[2].set_title(r'C. Phosphatidic Acid vs. Phospholipid Flux', fontsize=11, fontweight='bold')
    axes[2].set_xlabel(r'$\log_2$ Norm. PL-Flux [-]', fontsize=10)
    axes[2].set_ylabel(r'$\log_2$ Norm. PA Conc. [-]', fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Optimize layout geometry to prevent label/title clipping
    plt.tight_layout()
    plt.savefig('./results/sim_condition_variance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_fas_intermediates(df: pd.DataFrame) -> None:
    """
    Plots each FAS II intermediate pool inside its own individual subplot
    arranged in a structured 9x3 facet matrix, using rasterized curve parameters.
    """
    df_sorted = df.sort_values(by=PL_FLUX)
    x_normalized = log2_normalize_to_average(df_sorted, PL_FLUX)
    
    # 27 variables fit symmetrically into a 9 rows by 3 columns canvas matrix
    nrows, ncols = 9, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 15))
    
    for idx, (var, label, color) in enumerate(METABOLITES):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        if var in df_sorted.columns:
            ax.plot(
                x_normalized, 
                log2_normalize_to_average(df_sorted, var), 
                'o-', 
                markersize=3, 
                linewidth=1.2, 
                color=color, 
                alpha=0.85
            )
            ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='No Difference')
            ax.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.25)
            ax.tick_params(axis='both', labelsize=8)
            
            # Reduce visual clutter by only placing labels on the edges of the grid
            if row == nrows - 1:
                ax.set_xlabel(r'$\log_2$ Norm. PL-Flux [-]', fontsize=9)
            if col == 0:
                ax.set_ylabel(r'$\log_2$ Norm. Conc. [-]', fontsize=9)
        else:
            # If the column was discarded by the cleanup function, turn off the frame cleanly
            ax.text(0.5, 0.5, f'{var}\n[No Data]', ha='center', va='center', color='gray', transform=ax.transAxes)
            ax.set_axis_off()
            
    # Clean up empty spaces and avoid label overlapping
    plt.tight_layout()
    plt.savefig('./results/sim_fas_indep_flux_concs_grid.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_branch_fluxes(df: pd.DataFrame) -> None:
    """
    Plots 10 specific metabolic branch fluxes going into PLS and LPS synthesis 
    as a function of the log2 normalized phospholipid flux (vCdsA) in a 5x2 grid.
    Titles are centered and formatted using clear LaTeX biochemical notations.
    """
    # Define target fluxes along with their mapped colors and LaTeX formatted titles
    flux_configs = [
        ('v56PlsB', 'royalblue', r'v(C16:0-ACP)$_{\mathbf{PlsB}}$*'),
        ('v61PlsB', 'skyblue', r'v(C16:1-ACP)$_{\mathbf{PlsB}}$'),
        ('v66PlsB', 'royalblue', r'v(C18:0-ACP)$_{\mathbf{PlsB}}*$'),
        ('v71PlsB', 'skyblue', r'v(C18:1-ACP)$_{\mathbf{PlsB}}$*'),
        ('v76PlsB', 'royalblue', r'v(C20:0-ACP)$_{\mathbf{PlsB}}$'),
        ('v56PlsC', 'firebrick', r'v(C16:0-ACP)$_{\mathbf{PlsC}}$'),
        ('v61PlsC', 'salmon', r'v(C16:1-ACP)$_{\mathbf{PlsC}}$*'),
        ('v71PlsC', 'salmon', r'v(C18:1-ACP)$_{\mathbf{PlsC}}$*'),
        ('v81PlsC', 'salmon', r'v(C20:1-ACP)$_{\mathbf{PlsC}}$'),
        ('vLpxB', 'forestgreen', r'v$_{\mathbf{LpxB}}$')
    ]
    
    # Calculate the log2 normalized phospholipid flux axis
    x_data = log2_normalize_to_average(df, PL_FLUX)
    
    # Initialize a 5x2 canvas layout sharing x-axis limits
    fig, axes = plt.subplots(5, 2, figsize=(6, 12), sharex=True)
    axes = axes.flatten()
    
    for i, (flux_name, flux_color, latex_title) in enumerate(flux_configs):
        ax = axes[i]
        
        # Plot continuous line trajectory along with explicit sample points
        ax.plot(
            x_data, 
            log2_normalize_to_average(df, flux_name),
            'o-',
            color=flux_color, 
            linewidth=1.75,
            label=flux_name
        )
        
        # Center the LaTeX titles by removing the loc='left' parameter
        ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.set_title(latex_title, fontsize=12, pad=6, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', labelsize=9)
        
        # Conditional Labeling Constraint: Y-labels strictly on the left column (even indices)
        if i % 2 == 0:
            ax.set_ylabel(r'$\log_2$ Norm. Flux [-]', fontsize=10)
        else:
            ax.set_ylabel('') # Keep clean whitespace separation on the right column
            
        # X-labels restricted strictly to the bottom row plots (indices 8 and 9)
        if i >= 8:
            ax.set_xlabel(r'$\log_2$ Normalized PL-Flux [-]', fontsize=10)
            
    # Adjust spacing dynamically to prevent title/label overlap
    plt.tight_layout()
    plt.savefig('./results/sim_pls_branch_fluxes.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_lpa_accumulation(df_final: pd.DataFrame, df_nopi: pd.DataFrame) -> None:
    """
    Plots LPA (yP2) and PA (yP3) pools simultaneously against Mal-CoA using 
    dual y-axes to highlight comparative threshold bottlenecks.
    """
    # Sort both datasets independently to prevent plotting line zigzags
    df_f_sorted = df_final.sort_values(by='MalCoA')
    df_n_sorted = df_nopi.sort_values(by='MalCoA')
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    
    # --- AXIS 1 (Left): Lysophosphatidic Acid (LPA) ---
    ax1.set_xlabel(r'Mal-CoA Conc. [$\mu$M]', fontsize=10)
    ax1.set_ylabel(r'LPA Conc. [$\mu$M]', fontsize=10, color='royalblue')
    
    # LPA Control (darkturquoise) & LPA noPI (Cyan)
    ax1.plot(
        df_f_sorted['MalCoA'], df_f_sorted['yP2'], 
        'o-', color='darkturquoise', linewidth=1.5, markersize=4, alpha=0.85,
        label='LPA: Prod. Inhibition'
    )
    ax1.plot(
        df_n_sorted['MalCoA'], df_n_sorted['yP2'], 
        'o-', color='cyan', linewidth=1.5, markersize=4, alpha=0.85,
        label='LPA: No Prod. Inhibition'
    )
    
    # Format left axis ticks and spines to match the theme
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.spines['left'].set_color('royalblue')
    ax1.spines['left'].set_linewidth(1.5)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # --- AXIS 2 (Right): Phosphatidic Acid (PA) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'PA Conc. [$\mu$M]', fontsize=10, color='firebrick')
    
    # PA Control (orangered) & PA noPI (Gold)
    ax2.plot(
        df_f_sorted['MalCoA'], df_f_sorted['yP3'], 
        'o-', color='orangered', linewidth=1.5, markersize=4, alpha=0.85,
        label='PA: Prod. Inhibition'
    )
    ax2.plot(
        df_n_sorted['MalCoA'], df_n_sorted['yP3'], 
        'o-', color='gold', linewidth=1.5, markersize=4, alpha=0.85,
        label='PA: No Prod. Inhibition'
    )
    
    # Format right axis ticks and spines to match the theme
    ax2.tick_params(axis='y', labelcolor='firebrick')
    ax2.spines['right'].set_color('firebrick')
    ax2.spines['right'].set_linewidth(1.5)
    ax2.set_ylim(bottom=0)
    
    # --- Global Customization & Consolidated Legend ---
    ax1.set_xlim(0, 750)
    
    # Gather legends from both distinct axes into a single overlay box
    lines_left, labels_left = ax1.get_legend_handles_labels()
    lines_right, labels_right = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_left + lines_right, 
        labels_left + labels_right, 
        loc='center right', 
        fontsize=9, 
        framealpha=0.75
    )
    
    plt.tight_layout()
    plt.savefig('./results/sim_lpa_pa_accumulation.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_fa_composition(df: pd.DataFrame) -> None:
    """
    Plots the percentage proportion of different fatty acid intermediate fluxes 
    into phospholipid (PLS) and lipid A (LPS) synthesis pathways across all Mal-CoA conditions.
    
    Differentiates enzyme classes using geometric markers:
    - PlsB: Circles ('o')
    - PlsC: Squares ('s')
    - LpxB: Triangles ('^')
    """
    # Ensure sequential chronological tracing by sorting by the independent variable
    df_sorted = df.sort_values(by='MalCoA').reset_index(drop=True)
    
    # 1. Group flux entries cleanly by category to prepare for programmatic color assignment
    sat_fluxes = [
        ('v56PlsB', r'$v_{56\mathrm{PlsB}}$ (C16:0)*'),
        ('v66PlsB', r'$v_{66\mathrm{PlsB}}$ (C18:0)*'),
        ('v76PlsB', r'$v_{76\mathrm{PlsB}}$ (C20:0)'),
        ('v56PlsC', r'$v_{56\mathrm{PlsC}}$ (C16:0)'),
    ]
    
    unsat_fluxes = [
        ('v61PlsB', r'$v_{61\mathrm{PlsB}}$ (C16:1)'),
        ('v71PlsB', r'$v_{71\mathrm{PlsB}}$ (C18:1)*'),
        ('v61PlsC', r'$v_{61\mathrm{PlsC}}$ (C16:1)*'),
        ('v71PlsC', r'$v_{71\mathrm{PlsC}}$ (C18:1)*'),
        ('v81PlsC', r'$v_{81\mathrm{PlsC}}$ (C20:1)'),
    ]
    
    # 2. Programmatically Generate Color Gradients
    # Saturated Scale: Transitions from a bright sky tone, through royalblue, into a deep navy
    cmap_sat = mcolors.LinearSegmentedColormap.from_list("sat_gradient", ["#82B1FF", "royalblue", "#0A1931"])
    sat_colors = [cmap_sat(val) for val in np.linspace(0.0, 0.85, len(sat_fluxes))]
    
    # Unsaturated Scale: Transitions from a light coral pink, through firebrick, into a deep maroon
    cmap_unsat = mcolors.LinearSegmentedColormap.from_list("unsat_gradient", ["#FF8A80", "firebrick", "#3D0000"])
    unsat_colors = [cmap_unsat(val) for val in np.linspace(0.0, 0.85, len(unsat_fluxes))]
    
    # 3. Assemble Consolidated Configuration Map with Shape Coding
    # Form: (column_name, stoichiometric_weight, color, marker_shape, display_label)
    flux_configs = []
    
    for idx, (col, label) in enumerate(sat_fluxes):
        marker_shape = 's' if 'PlsC' in col else 'o'
        flux_configs.append((col, 1, sat_colors[idx], marker_shape, label))
        
    for idx, (col, label) in enumerate(unsat_fluxes):
        marker_shape = 's' if 'PlsC' in col else 'o'
        flux_configs.append((col, 1, unsat_colors[idx], marker_shape, label))
        
    # Append the isolated Lipid A tracking node (forced to forestgreen, a triangle marker, and weighted 4x)
    flux_configs.append(('vLpxB', 4, 'forestgreen', '^', r'$4 \times v_{\mathrm{LpxB}}$ (C14:0-OH)'))
    
    # 4. Compute the Cumulative Total Weighted Flux per Condition Profile Row
    total_weighted_flux = np.zeros(len(df_sorted))
    for col_name, weight, _, _, _ in flux_configs:
        total_weighted_flux += df_sorted[col_name] * weight
        
    # Safeguard against division-by-zero runtime exceptions
    total_weighted_flux = np.where(total_weighted_flux == 0, 1.0, total_weighted_flux)
    
    # 5. Build Canvas Elements
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for col_name, weight, flux_color, shape, flux_label in flux_configs:
        # Convert absolute metric scales into relative percentage properties
        percentage_flux = (df_sorted[col_name] * weight / total_weighted_flux) * 100
        
        ax.plot(
            df_sorted['MalCoA'], 
            percentage_flux, 
            linestyle='-',
            marker=shape, 
            color=flux_color, 
            linewidth=1.75, 
            markersize=5,  # Increased slightly to make squares/triangles pop cleanly
            alpha=0.85,
            label=flux_label
        )
    
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim(0, 750)
    ax.set_ylim(0, 50)  # Slight top pad to ensure 100% bounds don't clip lines
    ax.set_xlabel(r'Mal-CoA Conc. [$\mu$M]', fontsize=11)
    ax.set_ylabel(r'Normalized Flux [%]', fontsize=11)
    
    # Park the consolidated 10-item legend safely outside the axes grid bounds to prevent visual layout overlap
    ax.legend(loc='center right', fontsize=12, frameon=True, framealpha=0.8)
    
    # Render, adjust geometry, and save frame
    plt.tight_layout()
    plt.savefig('./results/sim_fa_utilization_composition.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_growth(df_final: pd.DataFrame) -> None:
    '''
    Calculates the growth rate using the linear approximation equation.
    '''
    N_A = 6.02214 * 10**(23-6)
    N_PL = 22 * 10**6
    V_cell = 3 * 10**-15

    growth_rates = df_final[PL_FLUX] * ((N_A * V_cell * 60**2) / N_PL)
    print(growth_rates)


def plot_acps(df: pd.DataFrame) -> None:
    """
    Plots the concentrations of free ACP (y4), the total saturated ACP pool,
    and the total unsaturated ACP pool as a function of Mal-CoA.
    Includes a reference line at y=48 to indicate the conserved total ACP pool.
    """
    # Sort data by MalCoA to guarantee continuous, non-overlapping lines
    df_sorted = df.sort_values(by='MalCoA').reset_index(drop=True)
    
    # Define saturated ACP intermediate variables (excluding freefattyacid profiles)
    saturated_acp_cols = [
        'y12', 'y13', 'y14', 'y15',  # C4
        'y17', 'y18', 'y19', 'y20',  # C6
        'y22', 'y23', 'y24', 'y25',  # C8
        'y27', 'y28', 'y29', 'y30',  # C10
        'y33', 'y34', 'y35', 'y36',  # C12
        'y43', 'y44', 'y45', 'y46',  # C14
        'y53', 'y54', 'y55', 'y56',  # C16
        'y63', 'y64', 'y65', 'y66',  # C18
        'y73', 'y74', 'y75', 'y76'   # C20
    ]
    
    # Define unsaturated ACP intermediate variables (excluding freefattyacid profiles)
    unsaturated_acp_cols = [
        'y32',                        # C10ucis
        'y38', 'y39', 'y40', 'y41',  # C12u
        'y48', 'y49', 'y50', 'y51',  # C14u
        'y58', 'y59', 'y60', 'y61',  # C16u
        'y68', 'y69', 'y70', 'y71',  # C18u
        'y78', 'y79', 'y80', 'y81'   # C20u
    ]
    
    # 1. Compute aggregate pool columns
    df_sorted['total_saturated_acp'] = df_sorted[saturated_acp_cols].sum(axis=1)
    df_sorted['total_unsaturated_acp'] = df_sorted[unsaturated_acp_cols].sum(axis=1)
    
    # 2. Initialize Canvas Layout
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 3. Plot the individual pool profiles
    # Free-ACP (y4) in forestgreen
    ax.plot(
        df_sorted['MalCoA'], df_sorted['y4'], 
        'o-', color='forestgreen', linewidth=2, markersize=4, alpha=0.9,
        label=r'Free ACP ($y_4$)'
    )

    # Mal-ACP (y10) in mediumvioletred
    ax.plot(
        df_sorted['MalCoA'], df_sorted['y10'], 
        'o-', color='hotpink', linewidth=2, markersize=4, alpha=0.9,
        label=r'Mal-ACP ($y_{10}$)'
    )
    
    # Total Saturated ACP Pool in royalblue
    ax.plot(
        df_sorted['MalCoA'], df_sorted['total_saturated_acp'], 
        'o-', color='royalblue', linewidth=2, markersize=4, alpha=0.9,
        label='Total Unb. Saturated ACPs'
    )
    
    # Total Unsaturated ACP Pool in firebrick
    ax.plot(
        df_sorted['MalCoA'], df_sorted['total_unsaturated_acp'], 
        'o-', color='firebrick', linewidth=2, markersize=4, alpha=0.9,
        label='Total Unb. Unsaturated ACPs'
    )
    
    # 4. Calculate and plot the live sum of all ACP-bound species
    total_acp_sum = df_sorted['y4'] + df_sorted['total_saturated_acp'] + df_sorted['total_unsaturated_acp'] + df_sorted['y10']
    
    ax.plot(
        df_sorted['MalCoA'], 
        total_acp_sum, 
        'o-',
        color='black', 
        linewidth=2, 
        zorder=1,
        label='Total Unb. ACP Pool',
        markersize=4
    )

    # 4. Add the black horizontal conservation line at y = 48
    ax.axhline(
        y=48, color='black', linestyle='--', linewidth=1.5, zorder=1,
        label=r'Total ACP Pool (48 $\mu$M)'
    ) 
    
    # Decorate plot labels and formatting grids
    ax.set_xlabel(r'Mal-CoA Conc. [$\mu$M]', fontsize=10)
    ax.set_ylabel(r'ACP Species Conc. [$\mu$M]', fontsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim(0, 750)
    ax.set_ylim(0, 50)  # Pad slightly above 48 to leave visual room for the line
    
    # Place legend
    ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.8)
    
    # Adjust layout boundaries and save the figure
    plt.tight_layout()
    plt.savefig('./results/sim_acp_pool.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_fas_elasticities(df: pd.DataFrame) -> None:
    """
    Calculates the local elasticity coefficients of FabG, FabZ, FabI, and FabF
    with respect to their direct pathway substrates between 75 uM and 150 uM Mal-CoA.
    
    Formula: epsilon = (ln(v_2) - ln(v_1)) / (ln(S_2) - ln(S_1))
    """
    # 1. Isolate the target data rows using a tolerance standard for floats
    row_75 = df[np.isclose(df['MalCoA'], 100.0, atol=1e-2)]
    row_150 = df[np.isclose(df['MalCoA'], 200.0, atol=1e-2)]
    
    if row_75.empty or row_150.empty:
        raise ValueError("Could not find exact rows matching Mal-CoA concentrations of 75 uM and 150 uM.")
        
    r1 = row_75.iloc[0]
    r2 = row_150.iloc[0]
    
    # Map enzymes to their specific ID numbers to keep f-strings clean
    enzyme_ids = {"FabG": 4, "FabZ": 5, "FabI": 6}
    
    # 2. Define standard lookup mappings for Saturated intermediates (Cycles 1 to 9)
    sat_cycles = range(1, 10)
    sat_sub_map = {
        'FabG': ['y12', 'y17', 'y22', 'y27', 'y33', 'y43', 'y53', 'y63', 'y73'],
        'FabZ': ['y13', 'y18', 'y23', 'y28', 'y34', 'y44', 'y54', 'y64', 'y74'],
        'FabI': ['y14', 'y19', 'y24', 'y29', 'y35', 'y45', 'y55', 'y65', 'y75'],
        'FabF': ['y15', 'y20', 'y25', 'y30', 'y36', 'y46', 'y56', 'y66']  # Length 8
    }
    
    # 3. Define standard lookup mappings for Unsaturated intermediates (Cycles 5 to 9)
    unsat_cycles = range(5, 10)
    unsat_sub_map = {
        'FabG': ['y38', 'y48', 'y58', 'y68', 'y78'],
        'FabZ': ['y39', 'y49', 'y59', 'y69', 'y79'],
        'FabI': ['y40', 'y50', 'y60', 'y70', 'y80'],
        'FabF': ['y32', 'y41', 'y51', 'y61']  # Length 4 (Starts at C10ucis, then C12u-C16u acyl)
    }
    
    results = []
    
    # --- Process Saturated System Configuration ---
    for enzyme in ['FabG', 'FabZ', 'FabI', 'FabF']:
        cycles = range(1, 8 + 1) if enzyme == 'FabF' else sat_cycles
        for idx, cycle in enumerate(cycles):
            # Cleaned up string formatting logic
            if enzyme == 'FabF':
                v_col = f'vcat8{cycle}'
            else:
                v_col = f'vcat{enzyme_ids[enzyme]}{cycle}'
                
            s_col = sat_sub_map[enzyme][idx]
            
            # Extract boundary condition measurements
            v1, v2 = r1[v_col], r2[v_col]
            s1, s2 = r1[s_col], r2[s_col]
            
            # Execute log-linear sensitivity math safely
            delta_ln_v = np.log(v2) - np.log(v1)
            delta_ln_s = np.log(s2) - np.log(s1)
            epsilon = delta_ln_v / delta_ln_s if delta_ln_s != 0 else np.nan
            
            results.append({
                'Enzyme': enzyme, 'Pathway': 'Saturated', 'Cycle': cycle,
                'Reaction_Col': v_col, 'Substrate_Col': s_col, 'Elasticity': epsilon
            })
            
    # --- Process Unsaturated System Configuration ---
    for enzyme in ['FabG', 'FabZ', 'FabI', 'FabF']:
        cycles = range(5, 8 + 1) if enzyme == 'FabF' else unsat_cycles
        for idx, cycle in enumerate(cycles):
            # Cleaned up string formatting logic (fixed the companion bug here too)
            if enzyme == 'FabF':
                v_col = f'vcat8un{cycle}'
            else:
                v_col = f'vcat{enzyme_ids[enzyme]}{cycle}d'
                
            s_col = unsat_sub_map[enzyme][idx]
            
            v1, v2 = r1[v_col], r2[v_col]
            s1, s2 = r1[s_col], r2[s_col]
            
            delta_ln_v = np.log(v2) - np.log(v1)
            delta_ln_s = np.log(s2) - np.log(s1)
            epsilon = delta_ln_v / delta_ln_s if delta_ln_s != 0 else np.nan
            
            results.append({
                'Enzyme': enzyme, 'Pathway': 'Unsaturated', 'Cycle': cycle,
                'Reaction_Col': v_col, 'Substrate_Col': s_col, 'Elasticity': epsilon
            })
            
    print(pd.DataFrame(results))


def main() -> None:
    print('INFO: Run this script from "model" folder for paths to work')
    df_FINAL = remove_unevaluated_data(pd.read_csv(PATH_FINAL_DATA))
    df_NOPI = remove_unevaluated_data(pd.read_csv(PATH_NOPI_DATA))
    plot_condition_variance_proof(df_FINAL, df_NOPI)
    plot_fas_intermediates(df_FINAL)
    plot_branch_fluxes(df_FINAL)
    plot_lpa_accumulation(df_FINAL, df_NOPI)
    plot_fa_composition(df_FINAL)
    plot_acps(df_FINAL)
    calculate_growth(df_FINAL)
    calculate_fas_elasticities(df_FINAL)


if __name__ == '__main__':
    main()
