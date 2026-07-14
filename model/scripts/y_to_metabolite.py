import sys, re
import pandas as pd


def _get_mapping(df: pd.DataFrame) -> dict[str, str]:
    ys = df.iloc[:, 0].tolist()
    names = df.iloc[:, 4].tolist()

    mapping = {y: name for y, name in zip(ys, names)}
    return mapping


def _to_metabolite_name(expr: str, mapping: dict[str, str]) -> str:
    pattern = re.compile(r"y\d{1,3}")

    def repl(match: re.Match) -> str:
        return mapping[match.group(0)]

    new_expr = pattern.sub(repl, expr.replace(' ', ''))
    new_expr = new_expr.replace('+', ' + ').replace('-', ' - ').replace('==', ' == ')
    return new_expr


def main(fpath_cons: str, fpath_mapp: str, fpath_out: str | None) -> None:
    print("==="*30)
    print(f"Conservation file: {fpath_cons}")
    print(f"Mapping file: {fpath_mapp}")
    print("==="*30, end="\n\n")

    df_cons: pd.DataFrame = pd.read_excel(fpath_cons)
    df_cons.columns = ['Y']
    print('Conservation DataFrame:')
    print(df_cons.head(), end='\n\n')

    df_mapp: pd.DataFrame = pd.read_excel(fpath_mapp, sheet_name="mapping")
    print('Mapping DataFrame:')
    print(df_mapp.head(), end='\n\n')

    df_cons['Name'] = df_cons.Y.apply(lambda expr: _to_metabolite_name(expr, _get_mapping(df_mapp)))
    
    if fpath_out is not None:
        print(f'Output file path specified: {fpath_out} ==> Saving...')
        with pd.ExcelWriter(fpath_out) as writer:
            df_cons[['Y']].to_excel(writer, sheet_name='Y', index=False, header=False)
            df_cons[['Name']].to_excel(writer, sheet_name='Name', index=False, header=False)
    else:
        print('Output file path not specified ==> Not saving!')
    print('Finished!')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Script takes 2 arguments! 1: path to conservations, 2: path to metadata/mapping...")
    if len(sys.argv) == 3:
        sys.argv += [None]
    
    main(*sys.argv[1:])
