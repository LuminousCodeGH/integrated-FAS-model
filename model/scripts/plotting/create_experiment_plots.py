'''
This script generates plots to help visualize the experimental data
It was made with the help of LLMs
'''
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr


PATH_DATA_ACP = './raw_output/experimental_lognorm_ACP_concentrations.csv'
PATH_DATA_ENZ = './raw_output/experimental_lognorm_enzyme_concentrations.csv'
PATH_DATA_PLS = './raw_output/experimental_lognorm_PL_concentrations.csv'
PATH_DATA_PLSDET = './raw_output/experimental_lognorm_PLdetailed_concentrations.csv'
PATH_OUTPUT = './results/'

SIGNIFICANTLY_VARYING_ENZ = ['FabF', 'FabG', 'FabI', 'FabZ', 'PlsC', 'CdsA']


def enzyme_concentration_plot(df_enz: pd.DataFrame) -> None:
    # 1. Map enzymes to their respective pathways
    pathway_map = {
        'FabA': 'FAS', 'FabB': 'FAS', 'FabD': 'FAS', 'FabF': 'FAS',
        'FabG': 'FAS', 'FabH': 'FAS', 'FabI': 'FAS', 'FabZ': 'FAS',
        'CdsA': 'PLS', 'PlsB': 'PLS', 'PlsC': 'PLS',
        'LpxC': 'LPS'
    }
    
    # Sort enzymes by pathway to group them visually in the grid
    enzymes = sorted(pathway_map.keys(), key=lambda x: (pathway_map[x], x))
    
    # 2. Define publication-quality color shades per pathway for [Batch 1, Batch 2]
    # FAS = Blues, PLS = Reds, LPS = Greens
    color_shades = {
        'FAS': ['royalblue', 'skyblue'],
        'PLS': ['firebrick', 'salmon'],
        'LPS': ['forestgreen', 'limegreen']
    } # https://htmlcolorcodes.com/color-names/
    
    # Get sorted unique batches dynamically
    batches = sorted(df_enz['batch'].unique())
    
    # 3. Initialize the 3x4 figure grid
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(16, 7))
    axes = axes.flatten()  # Flatten to a 1D array for easy 0-11 indexing
    
    # 4. Populate subplots
    for i, enz in enumerate(enzymes):
        ax = axes[i]
        pathway = pathway_map[enz]
        
        mean_col = f'{enz}_mean'
        std_col = f'{enz}_std'
        
        # Plot each batch sequentially
        for batch_idx, batch in enumerate(batches):
            # Subset data and isolate non-null concentrations for robustness
            batch_df = df_enz[df_enz['batch'] == batch].dropna(subset=[mean_col])
            
            # Skip if this batch completely lacks data for the current enzyme
            if batch_df.empty:
                continue
            
            # CRITICAL: Sort by x-axis so line plots connect sequentially, not erratically
            batch_df = batch_df.sort_values(by='log2_PL_flux_norm')
            
            # Select shade based on batch index
            color = color_shades[pathway][batch_idx % len(color_shades[pathway])]
            
            # Plot trends with error bars
            strain = batch_df['genotype'].iloc[0]
            ax.errorbar(
                batch_df['log2_PL_flux_norm'], 
                batch_df[mean_col], 
                yerr=batch_df[std_col], 
                fmt='o-', 
                color=color, 
                ecolor=color,
                capsize=3, 
                markersize=5,
                linewidth=1.5,
                label=f'Batch: {batch}',
                alpha=0.85
            )
        
        # Subplot Polish
        ax.set_title(enz + '*' if enz in SIGNIFICANTLY_VARYING_ENZ else enz, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.legend(fontsize=8, loc='best', frameon=True, facecolor='white', edgecolor='none')
        
        # Calculate row/col positions to isolate outer axis labels
        row = i // ncols
        col = i % ncols
        
        # Only label the lowest row's X-axis
        if row == nrows - 1:
            ax.set_xlabel(r'$\log_2$ Normalized PL Flux [-]', fontsize=10)
        
        # Only label the leftmost column's Y-axis (since sharey=True aligns them perfectly)
        if col == 0:
            ax.set_ylabel(r'$\log_2$ Norm. Enzyme Conc. [-]', fontsize=10)
            
    # 5. Figure Adjustments & Export
    plt.tight_layout()
    plt.savefig(f'{PATH_OUTPUT}exp_enzyme_concentrations_multiplot.png', dpi=300, bbox_inches='tight')
    plt.show()


def acylprotein_plot(df_acp: pd.DataFrame) -> None:
    # 1. Dynamically identify all ACP intermediate base names from headers
    mols = [col.replace('_mean', '') for col in df_acp.columns if col.endswith('_mean')]
    
    # 2. Define a sort key that groups by molecule type/suffix first, then carbon length
    def biochemical_sort_key(name):
        suffix_order = ['_0', '_0OH', '_0E', '_1Z', '_1Z_OH']
        
        if name == 'Malonyl':
            return (1, 0, name)
        if 'ACP' in name:
            return (1, 1, name)
            
        match = re.match(r'^C(\d+)(.*)', name)
        if match:
            carbons = int(match.group(1))
            suffix = match.group(2)
            # Find index in suffix_order; default to a high index if not found
            type_idx = suffix_order.index(suffix) if suffix in suffix_order else 2
            return (0, type_idx, carbons)
        return (2, 0, name)

    mols = sorted(mols, key=biochemical_sort_key)

    # 3. Extract molecule type for color grouping
    def get_molecule_type(name):
        if name in ['Malonyl', 'ACP_free', 'ACP_total']:
            return 'ACP' if 'ACP' in name else 'Malonyl'
        match = re.match(r'^C\d+(.*)', name)
        if match:
            return match.group(1)
        return 'Other'

    # 4. Color map configuration using standard Matplotlib/HTML names
    color_map = {
        '_0':      ['royalblue', 'skyblue'],  
        '_0OH':    ['firebrick', 'salmon'],         
        '_0E':     ['forestgreen', 'limegreen'],    
        '_1Z':     ['indigo', 'mediumpurple'],         
        '_1Z_OH':  ['orangered', 'orange'],      
        'Malonyl': ['mediumvioletred', 'hotpink'],     
        'ACP':     ['black', 'darkgray'],         
    }

    batches = sorted(df_acp['batch'].unique())
    
    # 5. Inverted Grid Dimensions for Portrait/Paper Layout (7 Rows x 5 Columns)
    nrows, ncols = 7, 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(16, 17))
    axes = axes.flatten()

    for i, mol in enumerate(mols):
        ax = axes[i]
        
        mol_type = get_molecule_type(mol)
        colors = color_map.get(mol_type, ['black', 'gray'])
        
        mean_col = f'{mol}_mean'
        std_col = f'{mol}_std'
        
        for batch_idx, batch in enumerate(batches):
            batch_df = df_acp[df_acp['batch'] == batch].dropna(subset=[mean_col])
            
            if batch_df.empty:
                continue
                
            batch_df = batch_df.sort_values(by='log2_PL_flux_norm')
            color = colors[batch_idx % len(colors)]
            print(f'Caption Legend: {color} = {batch} ')
            
            ax.errorbar(
                batch_df['log2_PL_flux_norm'],
                batch_df[mean_col],
                yerr=batch_df[std_col],
                fmt='o-',
                color=color,
                ecolor=color,
                capsize=2.5,
                markersize=4,
                linewidth=1.2,
                label=f'Batch {batch}',
                alpha=0.85
            )
        
        # 6. Nomenclature formatting rules for LaTeX safety
        if mol in ['Malonyl', 'ACP_free', 'ACP_total']:
            latex_title = mol.replace('_', '-')
        else:
            latex_title = mol.replace('_', ':', 1)
            latex_title = re.sub(r'(\d)OH', r'\1-OH', latex_title)
            latex_title = latex_title.replace('Z_OH', 'Z-OH')

        ax.set_title(latex_title, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # 7. Axis label constraints updated for row-major 7x5 layout
        row = i // ncols
        col = i % ncols
        
        # Only draw X-labels on the bottom-most row of subplots
        if row == nrows - 1:
            ax.set_xlabel(r'$\log_2$ Norm. PL Flux [-]', fontsize=10)
            
        # Only draw Y-labels on the left-most column of subplots
        if col == 0:
            ax.set_ylabel(r'$\log_2$ Norm. Concentration [-]', fontsize=10)

    # 8. Layout tuning & Saving
    plt.tight_layout()
    plt.savefig(f'{PATH_OUTPUT}exp_acp_intermediates_grid.png', dpi=300, bbox_inches='tight')
    plt.show()


def growth_plot(df_acp: pd.DataFrame, df_enz: pd.DataFrame, df_pls: pd.DataFrame) -> None:
    # 1. Helper function to safely isolate growth data by biological genotype
    def extract_growth_data(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
            
        mu_col = 'mu_mean' if 'mu_mean' in df.columns else 'mu'
        err_col = 'mu_error' if 'mu_error' in df.columns else 'mu_std'
        flux_col = 'log2_PL_flux_norm'
        genotype_col = 'genotype' 
        
        required = [genotype_col, flux_col, mu_col]
        if all(col in df.columns for col in required):
            sub_df = df[[genotype_col, flux_col, mu_col, err_col]].copy()
            sub_df = sub_df.rename(columns={mu_col: 'mu_mean', err_col: 'mu_error'})
            return sub_df
        return pd.DataFrame()

    # 2. Extract and pool the data from all three files
    data_sources = [
        extract_growth_data(df_acp),
        extract_growth_data(df_enz),
        extract_growth_data(df_pls)
    ]
    df_combined = pd.concat([df for df in data_sources if not df.empty], ignore_index=True)
    
    # 3. Drop missing entries and collapse redundant technical replicates (batches) per genotype
    df_combined = df_combined.dropna(subset=['log2_PL_flux_norm', 'mu_mean'])
    df_combined = df_combined.drop_duplicates(subset=['genotype', 'log2_PL_flux_norm'])
    
    # 4. Initialize canvas (slightly widened to gracefully fit the text-heavy legend entries)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    html_colors = ['darkturquoise', 'mediumorchid', 'orange']
    genotypes = sorted(df_combined['genotype'].unique())
    
    for idx, genotype in enumerate(genotypes):
        genotype_df = df_combined[df_combined['genotype'] == genotype]
        
        # Sort along X vector to ensure monotonic, continuous line plotting
        genotype_df = genotype_df.sort_values(by='log2_PL_flux_norm')
        color = html_colors[idx % len(html_colors)]
        
        x_data = genotype_df['log2_PL_flux_norm']
        y_data = genotype_df['mu_mean']
        
        # 5. Dynamic Pearson Correlation Calculation
        # Guard rails: verify there are enough distinct data coordinates to avoid division by zero
        if len(genotype_df) >= 3 and x_data.nunique() > 1 and y_data.nunique() > 1:
            r_coef, p_val = pearsonr(x_data, y_data)
            
            # Label annotation with significance flag
            sig_asterisk = '*' if p_val < 0.05 else ''
            legend_label = f'{genotype} (r = {r_coef:.2f}{sig_asterisk})'
        else:
            legend_label = f'{genotype} (insufficient data)'
        
        ax.errorbar(
            x_data,
            y_data,
            yerr=genotype_df['mu_error'],
            fmt='o-',
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
            linewidth=1.5,
            label=legend_label,
            alpha=0.85
        )
        
    # 5. Presentation Engineering & Typography
    ax.set_xlabel(r'$\log_2$ Normalized PL Flux [-]', fontsize=11)
    ax.set_ylabel(r'Specific Growth Rate $\mu$ [$\mathrm{h}^{-1}$]', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.legend(loc='best', frameon=True, fontsize=10)
    
    # 6. Layout Tuning & Save
    plt.tight_layout()
    plt.savefig(f'{PATH_OUTPUT}exp_growth_vs_phospholipid_flux.png', dpi=300, bbox_inches='tight')
    plt.show()


def main() -> None:
    df_ACP = pd.read_csv(PATH_DATA_ACP)
    df_ENZ = pd.read_csv(PATH_DATA_ENZ)
    df_PLS = pd.read_csv(PATH_DATA_PLS)
    df_PLSdet = pd.read_csv(PATH_DATA_PLSDET)

    enzyme_concentration_plot(df_ENZ)
    acylprotein_plot(df_ACP)
    growth_plot(df_ACP, df_ENZ, df_PLS)


if __name__ == '__main__':
    print('WARNING: Run from the "model" directory for paths to work!')
    main()
