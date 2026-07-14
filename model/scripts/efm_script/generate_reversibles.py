import numpy as np
import sys
from scipy.io import loadmat

# TesA is also irreversible, but it doesn't change the outcome of the EFM calculation
irrev_v = ['vcat1', 'vcat3', 'vcat4', 'vcat6', 'vcat7', 'vcat8', 'vcat10']  # Input the (ir)reversible rates here, make sure to ignore the following characters: ( ) _


if __name__ == '__main__':
    list_is_irrev = True
    if len(sys.argv) > 1:
        if sys.argv[1] == 'rev':
            list_is_irrev = False
            print('Matching reversible rates...')
        else:
            print('Matching irreversible rates...')
    else:
        print('Matching irreversible rates...')

    matrix_path = "./efm_script/input/S_matrix.mat"  # Run from the main 'model' folder!

    print('Loading matrix and v vector...')
    matdata = loadmat(matrix_path)
    S: np.ndarray = matdata["S"]
    metabs: np.ndarray = matdata["metabs"]
    v: np.ndarray = matdata["v"]
    arr = np.ones(len(v), dtype=int)

    print(f'(Ir)reversible Vs: {irrev_v}')
    print('Generating reversibilities...')
    for i in range(len(v)):
        if any(v[i].startswith(irv) for irv in irrev_v):
            print(f'{v[i]} found (ir)reversible!')
            arr[i] = 0
    
    arr = abs(arr - 1) if not list_is_irrev else arr
    print(f'Found {np.count_nonzero(arr)} reversible and {len(arr)-np.count_nonzero(arr)} irreversible reactions!')
    output: str = './efm_script/input/rev_reactions.npy'
    print(f'Saving reversibilities to {output}...')
    np.save(output, arr, False)
    print('Finished!')
