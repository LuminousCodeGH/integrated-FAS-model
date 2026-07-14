'''
This script generates plots to help visualize the optimization data
It was made with the help of LLMs
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


PATH_INDIV_DATA = './raw_output/evolution_metrics_individual_PlsBbeta.csv'


def plot_evolution(df_indiv: pd.DataFrame) -> None:
    """
    Plots the evolutionary optimization trajectory using twin y-axes:
    - Left Axis (Black/Firebrick): Beta parameter distribution and best individual Beta.
    - Right Axis (Royal Blue): The historic best individual's average flux output performance.
    """
    df = df_indiv.copy()
    
    # Ensure dataframe is sequentially ordered by generation
    df = df.sort_values('ApproxGeneration').reset_index(drop=True)
    
    # 1. Group by generation to compute the mean and standard deviation of the Beta parameter
    gen_stats = df.groupby('ApproxGeneration')['Beta'].agg(['mean', 'std']).reset_index()
    
    # 2. Locate the best individual row *within* each generation first
    best_per_gen_idx = df.groupby('ApproxGeneration')['AvgFluxOutput'].idxmax()
    df_gen_bests = df.loc[best_per_gen_idx].sort_values('ApproxGeneration').reset_index(drop=True)
    
    # 3. Compute the cumulative historic best individual's Beta AND Flux values over generations
    best_overall_betas = []
    best_overall_fluxes = []
    max_flux_so_far = -np.inf
    best_beta_so_far = None
    
    for _, row in df_gen_bests.iterrows():
        # Update the champion profile only if an absolute higher flux output is achieved
        if row['AvgFluxOutput'] > max_flux_so_far:
            max_flux_so_far = row['AvgFluxOutput']
            best_beta_so_far = row['Beta']
        best_overall_betas.append(best_beta_so_far)
        best_overall_fluxes.append(max_flux_so_far)
        
    df_gen_bests['best_overall_beta'] = best_overall_betas
    df_gen_bests['best_overall_flux'] = best_overall_fluxes
    gen_best = df_gen_bests[['ApproxGeneration', 'best_overall_beta', 'best_overall_flux']]
    
    # Combine stats for plotting
    plot_data = pd.merge(gen_stats, gen_best, on='ApproxGeneration')
    
    # Initialize the plot layout
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # --- LEFT AXIS: Parameter Metrics (Beta) ---
    line_avg = ax1.plot(
        plot_data['ApproxGeneration'], 
        plot_data['mean'], 
        color='black', 
        linewidth=2, 
        label=r'Generation Average $\beta$'
    )[0]
    
    ax1.fill_between(
        plot_data['ApproxGeneration'],
        plot_data['mean'] - plot_data['std'],
        plot_data['mean'] + plot_data['std'],
        color='black',
        alpha=0.12,
        label=r'Generation Variance ($\pm$1 SD)'
    )
    
    line_best_beta = ax1.plot(
        plot_data['ApproxGeneration'], 
        plot_data['best_overall_beta'], 
        color='firebrick', 
        linewidth=2.5, 
        label=r'Best Overall Individual $\beta$'
    )[0]
    
    ax1.set_xlabel('Generation [-]', fontsize=10)
    ax1.set_ylabel(r'Parameter Value $\beta$ [-]', color='black', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # --- RIGHT AXIS: Fitness Metrics (Flux Output) ---
    ax2 = ax1.twinx()
    line_best_flux = ax2.plot(
        plot_data['ApproxGeneration'], 
        plot_data['best_overall_flux'],
        ':',
        color='royalblue', 
        linewidth=2.5, 
        label='Best Overall Flux Output',
    )[0]
    
    ax2.set_ylabel(r'Flux Output [$\frac{\mu \mathrm{M}}{s}$]', color='royalblue', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='royalblue')
    
    # Match the right axis spine color to the royalblue data line
    ax2.spines['right'].set_color('royalblue')
    ax1.spines['left'].set_color('black')
    
    # --- UNIFIED LEGEND ---
    # Gather handles and labels from both coordinate systems into a single legend frame
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='lower right', frameon=True)
    
    # Clean up bounds and titles
    ax1.set_xlim(0, 102)
    ax1.set_title(r'Optimization of PlsB $\beta$ using DifferentialEvolution', fontsize=12, fontweight='bold')
    
    # Adjust layout and export
    plt.tight_layout()
    plt.savefig('./results/sim_plsb_beta_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()


def main() -> None:
    df_INDIV = pd.read_csv(PATH_INDIV_DATA)
    plot_evolution(df_INDIV)    


if __name__ == '__main__':
    main()
