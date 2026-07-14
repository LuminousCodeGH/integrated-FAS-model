# Integrated Dynamic Model: FAS, LPS, and PLS

Welcome to the public repository for our master's thesis project Systems Biology at the Vrije Universiteit Amsterdam. 

This repository contains the code, data, and scripts used to model and analyze three interconnected biosynthetic pathways: **Fatty Acid Synthesis (FAS)**, **Lipopolysaccharide (LPS) Synthesis**, and **Phospholipid Synthesis (PLS)**. 

It was used to study how growth perturbations affect the membrane phospholipid composition.

---

## Model Attributions & Credits

This integrated model stands on the shoulders of existing work. Specifically:
*   **Fatty Acid Synthesis (FAS):** The FAS pathway kinetics are adapted and built upon the model by [**Ruppe et al. (2020)**](https://doi.org/10.1073/pnas.2013924117). (Referenced in our files and data as the "Fox" model).
*   **Lipopolysaccharide (LPS) Synthesis:** The LPS pathway kinetics are adapted from the model by [**Emiola et al. (2015)**](https://doi.org/10.1371/journal.pone.0121216).
*   **Experimental Data:** Used to compare our simulated steady-state concentrations to by [**Noga et al. (2020)**](https://doi.org/10.1128/mbio.02703-19).

---

## System Requirements & Dependencies

To run the models and scripts in this repository, we used the following environment:

*   **Mathematica:** Version 14.3
*   **Python:** Version 3.12.3

### Python Packages
You can install the exact versions used during development by saving the block below as a `requirements.txt` file, or simply reference them here:

```text
numpy==1.26.4
pandas==2.1.4
matplotlib==3.6.3
scipy==1.11.4
```

To install these dependencies all at once, run this command in your terminal:

```bash
pip install -r requirements.txt
```

---

## Repository Structure

Here is a map of how the files are organized:

### 1. Main Modeling Notebooks (Root Directory)
These Wolfram Mathematica (`.nb`) notebooks are the core of the project.
*   `NewModel.nb`: The primary notebook containing the multi-pathway model for steady state concentration simulations.
*   `NewModelOptimization.nb`: The optimization of the product inhibition parameter.
*   `NewModelStoichioCheck.nb`: Stoichiometric analysis using a package for the new model.
*   `FoxModel.nb`: The model for the FAS, adapted from MATLAB.
*   `FoxModelStoichioCheck.nb`: Stoichiometric analysis using a package for the FAS model.

### 2. Model Data (`/data`)
All the parameters, equations, and experimental values that feed into the simulations.
*   `FAS/`: Kinetic parameters, total enzyme concentrations, import/export equations, and ODE layouts (for both 2020 and 2022 versions of the reference model) in CSV and Excel formats.
*   `LPS/`: ODEs, rates, and enzyme concentrations for the Lipopolysaccharide pathway.
*   `PLS/`: ODEs, rates, and enzyme concentrations for the Phospholipid pathway.
*   `new_model/`: Initial concentration values (`inits`) and ODE setups specifically formatted for our integrated model.
*   `experimental/`: Log-normal distribution datasets for ACP, PL (phospholipids), and various enzyme concentrations.
*   `shared/`: Kinetic parameters for Michaelis-Menten (MM) reactions and fixed concentration files for the FAS, LPS and PLS pathways.

### 3. Utility & Helper Scripts (`/scripts`)
A mix of Python and Mathematica scripts used for data conversion, formatting, parameter determination, and analysis.
*   `MATLAB_to_Mathematica.nb`: Converts raw MATLAB ODE structures into a format Mathematica can easily parse.
*   `combine_branches.nb`: Helper script used to programmatically merge the different pathway branches and check their stoichiometry.
*   `mass_action_in_rates.py` & `y_to_metabolite.py`: Python utility scripts for reexpressing ODEs and printing metabolite names.
*   `enzyme_concentration_estimation.nb` & `malonyl_Km_determination.nb`: Dedicated notebooks for estimating enzyme concentrations and substrate affinities.
*   `efm_script/`: Contains Python tools (`calculate_efms.py`, `generate_reversibles.py`) and MATLAB matrices (`.mat`) to compute **Elementary Flux Modes (EFMs)**.
*   `plotting/`: Python scripts to visualize simulated and experimental data.
*   `MM_LPS_exports.nb` & `MM_PLS_exports.nb` : Notebooks containing the rates and ODEs for the branching pathways. Edit equations here!

### 4. Input, Output & Results
*   `metadata/`: Includes `variable_mappings.xlsx` to help you map variable names across different datasets.
*   `raw_input/`: The original, unaltered MATLAB ODE representations we imported.
*   `raw_output/`: Unprocessed simulation outputs, including genetic algorithm evolution metrics for optimizing PlsB beta.
*   `results/`: Saved visualizations from our runs.

---

## How to Get Started

### Working with the Models (Mathematica)
1. Ensure you have Wolfram Mathematica installed (Version 14.3 recommended).
2. Check the equations for the branches (`MM_[BRANCH]_exports.nb`) and generate branch equations.
3. Run the `combine_branches.nb` script to merge the equations from the FAS, LPS and PLS.
4. Run the simulation notebooks in the `model` folder.

### Running the Python Scripts
If you want to run the Python scripts:
1. Install the necessary Python packages listed above.
2. Run the plotting scripts inside `scripts/plotting/`, the EFM calculators in `scripts/efm_script/`, or others in `scripts/` always from the main `model` folder.

---

## Known Bugs/Issues
*   EFM calculation in `efm_script/` takes too long/freezes due to combinatorial explosion.
*   When saving best in generation to the Excel in `NewModelOptimization.nb`, the best is compared against the parameter instead of fitness causing the tracking to stop if the parameter decreases. Use individuals dataset instead!
