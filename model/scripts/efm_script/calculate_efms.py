import argparse
import efmtool as efm
import pandas as pd
import numpy as np

from scipy.io import loadmat, savemat


testdata = {"S": np.array([[1, 1, -1, 0, -1],
                           [0, 0, 1, -1, 0],
                           [0, 0, 0, 1, -1]]),
            "metabs": np.array(['A', 'B', 'C']),
            "v": np.array(['v1', 'v2', 'v3', 'v4', 'v5'])}


def _calculate_efms(matrix: np.ndarray, 
                    metab_vars: np.ndarray, 
                    react_vars: np.ndarray, 
                    rev_list: np.ndarray=None) -> np.ndarray:
    if rev_list is None:
        rev_list = np.array([1] * matrix.shape[1])  # Make the reactions all reversible (same as normal nullspace calculation!)
    print(f'Calculating with {np.count_nonzero(rev_list)} reversible and {len(rev_list)-np.count_nonzero(rev_list)} irreversible reactions...')
    efm_res: np.ndarray = efm.calculate_efms(matrix, rev_list, react_vars, metab_vars)

    print(f"Finished calculating EFMs, resulting matrix is of dimensions: {efm_res.shape}")
    return efm_res.transpose()[np.argsort(efm_res.transpose().sum(axis=1))]


def convert_to_conservations(matrix: np.ndarray, v: np.ndarray) -> np.ndarray:
    import sympy as sp
    print('Converting to conservations...')
    v_symbols = sp.symbols(' '.join(v))
    equations_l: list[sp.Expr] = sp.Matrix(matrix) * sp.Matrix(v_symbols)
    equations = np.array([str(eq).replace('1.0*', '').replace('.0', '').replace('  ', ' ') + ' == Constant' for eq in equations_l])
    print(f'Example: {equations[0]} (total shape: {equations.shape})')
    return equations


def save_efm_relations(equations: np.ndarray, output_path: str, name:str='EFMs') -> None:
    print(f'Saving EFM conservations to {output_path}...')
    df = pd.DataFrame({name: equations})
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name=name, index=False)


def save_matrix_to_disk(matrix: np.ndarray, v: np.ndarray, metabs: np.ndarray, matrix_name: str, matrix_path: str, output_path: str) -> None:
    mat_output = '/'.join(output_path.split('/')[:-1]) + '/' + matrix_path.split('/')[-1].replace('S', matrix_name)
    print(f'Saving {matrix_name}-matrix to {mat_output}...')
    if matrix_name == 'K':
        data_to_save = {
            'K': matrix,
            'columns': v
        }
    elif matrix_name == 'L0':
        data_to_save = {
            'L0': matrix,
            'columns': metabs
        }
    
    savemat(mat_output, data_to_save)


def main(matrix_path: str, 
         rev_reactions_path: str,
         output_path: str,
         metabolites: bool,
         save_matrix: bool) -> None:
    matdata = testdata
    rev_reactions = None
    if matrix_path != "test":
        matdata = loadmat(matrix_path)
        rev_reactions = np.load(rev_reactions_path)
    S: np.ndarray = matdata["S"]
    metabs: np.ndarray = matdata["metabs"]
    v: np.ndarray = matdata["v"]

    print(f'Loaded S-matrix with dimensions: {S.shape} ({matrix_path})', end=' ==> ')
    assert len(metabs) == S.shape[0] and len(v) == S.shape[1], 'Row or column names are of different length: {} x {}'.format(len(metabs), len(v))
    print(f'Length matches: {len(metabs)} x {len(v)}')
    print(f'v: {v[:5]}')
    print(f'metabs: {metabs[:5]}')

    if metabolites:
        matrix = _calculate_efms(S.T, v, metabs, np.array([0] * S.T.shape[1]))
        matrix_name = 'L0'
        equations = convert_to_conservations(matrix, metabs)
        save_efm_relations(equations, output_path, name='ECRs')
    else:
        matrix = _calculate_efms(S, metabs, v, rev_reactions)
        matrix_name = 'K'
        equations = convert_to_conservations(matrix, v)
        save_efm_relations(equations, output_path)

    if save_matrix:
        save_matrix_to_disk(matrix, v, metabs, matrix_name, matrix_path, output_path)
    print('Finished sucessfully!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate elementary flux modes (EFMs)."
    )

    parser.add_argument(
        "matrix_path",
        nargs="?",
        default="./efm_script/input/S_matrix.mat",
        help="Path to S-matrix .mat file, or 'test' for built-in test data"
    )

    parser.add_argument(
        "rev_reactions_path",
        nargs="?",
        default="./efm_script/input/rev_reactions.npy",
        help="Path to reversible reactions .npy file"
    )

    parser.add_argument(
        "--output",
        default="./data/fox_EFM_conservations.xlsx",
        help="Output Excel file for conservation relations"
    )

    parser.add_argument(
        "--metabolites",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Transpose matrix or not to get metabolite conservations"
    )

    parser.add_argument(
        "--save_matrix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the resulting coefficient matrix as a *.mat"
    )

    args = parser.parse_args()

    main(
        args.matrix_path,
        args.rev_reactions_path,
        args.output,
        args.metabolites,
        args.save_matrix
    )
