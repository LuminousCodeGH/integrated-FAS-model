import sys
import re
import pandas as pd


test_df: pd.DataFrame = pd.DataFrame({'Eq': [
    'dy(1,:) = k1_1r.*y(83,:) - k1_1f.*e1.*y(1,:)',
    ' dy(4,:) = k2_3r.*y(89,:) - k2_3f.*y(88,:).*y(4,:)+ kcat7(1).*y(101,:)+ kcat7(2).*y(116,:) + kcat7(3).*y(131,:) + kcat7(4).*y(146,:) + kcat7(5).*y(161,:) + kcat7(6).*y(176,:) + kcat7(7).*y(191,:) + kcat7(8).*y(206,:) + kcat7(9).*y(221,:) + kcat7(5).*y(230,:) + kcat7(6).*y(245,:) + kcat7(7).*y(260,:) + kcat7(8).*y(275,:) + kcat7(9).*y(290,:) + k8_2f(1).*y(102,:) - k8_2r(1).*y(103,:).*y(4,:) + k8_2f(2).*y(117,:) - k8_2r(2).*y(118,:).*y(4,:) + k8_2f(3).*y(132,:) - k8_2r(3).*y(133,:).*y(4,:) + k8_2f(4).*y(147,:)- k8_2r(4).*y(148,:).*y(4,:) + k8_2f(5).*y(162,:)- k8_2r(5).*y(163,:).*y(4,:) + k8_2f(6).*y(177,:)- k8_2r(6).*y(178,:).*y(4,:) + k8_2f(7).*y(192,:)- k8_2r(7).*y(193,:).*y(4,:) + k8_2f(8).*y(207,:)- k8_2r(8).*y(208,:).*y(4,:) + k8_2f(5).*y(231,:)- k8_2r(5).*y(232,:).*y(4,:) + k8_2f(6).*y(246,:)- k8_2r(6).*y(247,:).*y(4,:) + k8_2f(7).*y(261,:)- k8_2r(7).*y(262,:).*y(4,:) + k8_2f(8).*y(276,:)- k8_2r(8).*y(277,:).*y(4,:) + k10_2f(1).*y(107,:) - k10_2r(1).*y(108,:).*y(4,:) + k10_2f(2).*y(122,:) - k10_2r(2).*y(123,:).*y(4,:) + k10_2f(3).*y(137,:) - k10_2r(3).*y(138,:).*y(4,:) + k10_2f(4).*y(152,:)- k10_2r(4).*y(153,:).*y(4,:) + k10_2f(5).*y(167,:)- k10_2r(5).*y(168,:).*y(4,:) + k10_2f(6).*y(182,:)- k10_2r(6).*y(183,:).*y(4,:) + k10_2f(7).*y(197,:)- k10_2r(7).*y(198,:).*y(4,:) + k10_2f(8).*y(212,:)- k10_2r(8).*y(213,:).*y(4,:) + k10_2f(5).*y(236,:)- k10_2r(5).*y(237,:).*y(4,:) + k10_2f(6).*y(251,:)- k10_2r(6).*y(252,:).*y(4,:) + k10_2f(7).*y(266,:)- k10_2r(7).*y(267,:).*y(4,:) + k10_2f(8).*y(281,:)- k10_2r(8).*y(282,:).*y(4,:) + k10_2f(4).*y(302,:)- k10_2r(4).*y(303,:).*y(4,:) + k3_inh_r.*y(293,:)  - k3_inh_f.*e3.*y(4,:) + k4_inh_r.*y(294,:)  - k4_inh_f.*e4.*y(4,:) + k5_inh_r.*y(295,:)  - k5_inh_f.*e5.*y(4,:) + k6_inh_r.*y(296,:)  - k6_inh_f.*e6.*y(4,:) + k7_inh_r.*y(297,:)  - k7_inh_f.*e7.*y(4,:) + k8_inh_r.*y(298,:)  - k8_inh_f.*e8.*y(4,:) + k9_inh_r.*y(299,:)  - k9_inh_f.*e9.*y(4,:) + k10_inh_r.*y(300,:)  - k10_inh_f.*e10.*y(4,:)',
    ' dy(6,:) = k6_1r(1).*y(94,:) - k6_1f(1).*e6.*y(6,:)',
    ' dy(7,:) = kcat1_2.*y(86,:)',
    ' dy(8,:) = kcat1_2.*y(86,:) + k2_1r.*y(87,:) - k2_1f.*e2.*y(8,:)'
]})


reverse_cats: dict[str, str] = {
    'kcat5': 'kcat5f',
    'k5_2r': 'kcat5r',
    'kcat9': 'kcat9f',
    'k9_2r': 'kcat9r'
}


def matlab_to_sympy(eq: str) -> str:
    '''Converts Matlab-style equations to equations Sympy can read for each row in the DataFrame'''
    eq = eq.replace('dy', 'y').replace('y(', 'y').replace(',:)', '').replace('.*', '*')
    return eq


def sympy_to_mathmatica(eq: str) -> str:
    '''Converts Sympy-style equations to Mathematica-style equations for each row in the DataFrame'''
    eq = eq.replace('*', ' ')
    eq = re.sub(r'\b(e\d+)\b', r'\1[t]', eq)  # Replace values of 'e1', 'e10', etc. with 'e1[t]', 'e10[t]'
    eq = re.sub(r'\b(y\d+)\b', r'\1[t]', eq)  # Replace values of 'y1', 'y10', etc. with 'y1[t]', 'y10[t]'
    return eq


def get_subs_map(d: dict[str, str]) -> dict[str, str]:
    l = len(d)
    new_d = {v: k for k, v in d.items()}
    assert l == len(new_d), 'Length of the normal dict and the substitution map do not match!'
    return new_d


def get_fwd_bwd_rates(rhs: str, rates_fb: dict[str, str]) -> None:
    '''Get the forward an backward rates from the right-hand side'''
    exprs_fb: list[str] = rhs.replace('-', '+').split('+')
    for expr in exprs_fb:
        terms: list[str] = expr.split('*')
        rate_fb_name = ''
        for term in terms:
            if term.startswith('k'):
                if rate_fb_name:
                    raise ValueError(f'Two kinetic parameters are defined in this expression! ({expr})')
                # If the term matches the name in the reverse catalyzer list, change the name for consistency
                for k in reverse_cats.keys():
                    if k in term:
                        print(f'Found {term}, replacing {k} with {reverse_cats[k]}...')
                        term = term.replace(k, reverse_cats[k])
                rate_fb_name = term.replace('k', 'v')
        while rates_fb.get(rate_fb_name, expr) != expr:
            # While there exists an entry for the rate name that does not match in expression, add a 'd' to the name (for duplicate)
            _old_expr = rates_fb[rate_fb_name]
            print(f'WARNING: Rate \'{rate_fb_name}\' has a different expression associated with it: \'{_old_expr}\' (OLD) / \'{expr}\' (NEW) ==> Making unique...')
            rate_fb_name += 'd'
        rates_fb.update({rate_fb_name: expr})


def get_net_rates(rates_fb: dict[str, str], in_fb_rates=True) -> dict[str, str]:
    '''Get the net rates from by combining the forward and backward rates'''
    rates_net = {r.replace('f', '').replace('r', ''): '' for r in rates_fb.keys()}
    for k, v in rates_fb.items():
        rate_net_name = k.replace('f', '').replace('r', '')
        term = k if in_fb_rates else v
        if 'r' in k:
            rates_net[rate_net_name] = f'{rates_net[rate_net_name]}-{term}'.replace('+-', '-').strip('+')
        else:
            rates_net[rate_net_name] = f'{term}+{rates_net[rate_net_name]}'.replace('+-', '-').strip('+')
    
    return rates_net


def substitute_eq_with(rhs: str, rates_fb: dict[str, str], subs_with_net=False) -> str:
    '''Substitute the right-hand side with the rates (fwd/bwd or net) row by row in a DataFrame'''
    subs_map_fb = get_subs_map(rates_fb)

    # Magic regex to replace longest matching substring from the subs_map
    pattern = re.compile('|'.join(map(re.escape, sorted(subs_map_fb, key=len, reverse=True))))
    result_fb = pattern.sub(lambda m: subs_map_fb[m.group()], rhs)

    if not subs_with_net:
        return result_fb
    
    # The order and sign of the terms in may not match the definition of the rates in the dict, so just subsitution wouldn't work
    rates_net = get_net_rates(rates_fb, in_fb_rates=True)
    rate_list_fb = result_fb.replace('+', ' ').replace('-', ' -').split(' ')
    # Since we know all names are unique and there are no constants aside from +1 and -1, we can use a set to autoremove duplicates 
    replace_set: set[str] = set()
    for rate in rate_list_fb:
        if ('f' in rate and rate.startswith('-')) or ('r' in rate and not rate.startswith('-')):
            # Reversed reaction
            rate_net = rate.replace('f', '').replace('r', '').replace('-', '')
            assert rate_net in rates_net.keys(), 'Reversed net rate \'{}\' not found in dictionary!'.format(rate_net)
            replace_set.add(f'-{rate_net}')
        elif ('f' in rate and not rate.startswith('-')) or ('r' in rate and rate.startswith('-')):
            # Normal direction reaction
            rate_net = rate.replace('f', '').replace('r', '').replace('-', '')
            assert rate_net in rates_net.keys(), 'Net rate \'{}\' not found in dictionary!'.format(rate_net)
            replace_set.add(rate_net)
        else:
            # Not a fwd/bwd rate, but catalytic or something like that
            replace_set.add(rate)
    result_net = '+'.join(replace_set).replace('+-', '-')

    return result_net


def dict_to_dataframe(d: dict[str, str]) -> pd.DataFrame:
    new_d = {'Rate': [], 'Expression': []}
    for k, v in d.items():
        new_d['Rate'].append(k)
        new_d['Expression'].append(v)
    return pd.DataFrame(new_d)


def save_ODEs_in_rates(df: pd.DataFrame, rates_fb: dict[str, str], output_path: str) -> None:
    '''Saves the updated DataFrames to an Excel file'''
    print('Generating rate mappings from dictionaries...')
    df_fb = dict_to_dataframe(rates_fb)
    df_fb.Expression = df_fb.Expression.apply(sympy_to_mathmatica)
    df_net = dict_to_dataframe(get_net_rates(rates_fb, in_fb_rates=True))
    df_net.Expression = df_net.Expression.apply(sympy_to_mathmatica)

    df.Eq = df.Eq.apply(sympy_to_mathmatica)

    print('Writing three Excel sheets...')
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name='ODEs', index=False)
        df_fb.to_excel(writer, sheet_name='fb_rates', index=False)
        df_net.to_excel(writer, sheet_name='net_rates', index=False)

    print(f"Saved updated ODEs and rate mappings to '{output_path}'")


def main(fpath: str) -> None:
    '''Takes an Excel sheet with mass action ODEs and converts the right-hand side to rates'''
    df = test_df
    d_rates_fb: dict[str, str] = {}
    is_test = True
    if fpath:
        is_test = False
        print(f'Loading ODE data from: {fpath}...')
        if fpath.endswith('.xls') or fpath.endswith('.xlsx'):
            df = pd.read_excel(fpath, header=None)
        elif fpath.endswith('.csv'):
            df = pd.read_csv(fpath, header=None)
        else:
            raise ValueError('File path does not specify a file format!')
        df.columns = ['Eq']  # The original Excel does not have headers, define them here
        model_version = fpath.split('/')[-1].split('_')[0]
    # Clean the equations and parse the equations to a format Sympy can easily read
    df.Eq = df.Eq.str.replace(' ', '')
    df.Eq = df.Eq.apply(lambda eq: matlab_to_sympy(eq))
    # Split into right and left side and get the terms from the RHS
    df[['LHS', 'RHS']] = df['Eq'].str.split('=', expand=True)
    print('Calculating the forward and backward rates...')
    df.RHS.apply(lambda rhs: get_fwd_bwd_rates(rhs, d_rates_fb))
    # Get the net rates by combining the fwd/bwd rates
    print('Calculating the net rates...')
    d_rates_net = get_net_rates(d_rates_fb, in_fb_rates=True)
    print(f'Total fwd/bwd rates: {len(d_rates_fb)}')
    print(f'Total net rates: {len(d_rates_net)}')

    if is_test:
        print('')
        print(d_rates_fb)
        print(d_rates_net)
        print('')
        assert len(d_rates_net) == 52 and len(d_rates_fb) == 89, 'TEST FAILED: Number of rates do not align! ==> Should be 89 and 52'
    
    print('Substituting main ODE...')
    df['RatesFB'] = df.RHS.apply(lambda rhs: substitute_eq_with(rhs, d_rates_fb, subs_with_net=False))
    df['RatesNet'] = df.RHS.apply(lambda rhs: substitute_eq_with(rhs, d_rates_fb, subs_with_net=True))

    df['LHS'] = df['LHS'].apply(sympy_to_mathmatica)
    df['EqFB'] = df['LHS'] + ' = ' + df['RatesFB']
    df['EqNet'] = df['LHS'] + ' = ' + df['RatesNet']

    print(f"Forward-backward rates:\n{df['EqFB'].head()}")
    print(f"\nNet rates:\n{df['EqNet'].head()}\n")

    if is_test:
        print('This is a test run... not saving the output! ==> Finished!')
        return
    
    output_path = '/'.join(fpath.split('/')[:-1] + [f'{model_version}_fox_ODEs_in_rates.xlsx'])
    print(f'Saving DataFrames to file: {output_path}')
    save_ODEs_in_rates(df[['Eq', 'EqFB', 'EqNet']].copy(True), d_rates_fb, output_path)


if __name__ == "__main__":
    fpath = ''
    if len(sys.argv) < 2:
        print('No arguments detected ==> Using test DataFrame!')
    else:
        fpath = sys.argv[1]
    main(fpath)
