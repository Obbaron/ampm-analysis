import pandas as pd
from pathlib import Path


def get_parts(filepath : str | Path, parametric : bool = False) -> pd.DataFrame:
    """
    Create pd.DataFrame from parts data exported from QuantAM (.csv)

    Parameters:
    -----------
    filepath : str or Path
        Filepath to parts data file (.csv)
    parametric : bool, optional
        Include laser parameters in output (default: False)

    Returns
    -----------
    part_data : pd.DataFrame
        DataFrame of Part ID, Layer Thickness, X Position, Y Position, Layers Count
        Optionally include Hatch Power, Hatch Point Distance, Hatch Exposure Time
    """

    filepath = Path(filepath) if type(filepath) == str else filepath
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    parts_data = pd.read_csv(filepath,
                    usecols=[1,3,4,5,6],
                    names=['Part ID','Layer Thickness','X Position','Y Position','Layers Count'],
                    skiprows=6,
                    on_bad_lines='skip',
                    skip_blank_lines=True
                    )
    
    if 'Tab - 10' in parts_data['Part ID'].values:
        parametric_possible = True
    else:
        parametric_possible = False
        
    parts_idx = []
    for value in parts_data['Part ID']:
        if pd.isna(value):
            break
        if isinstance(value, str) and value.startswith('Tab - '): # Exports from QuantAM go straight to next tab
            break
        parts_idx.append(int(value)-1)

    parts_data = parts_data.loc[parts_idx]
    
    
    if parametric and not parametric_possible:
        raise ValueError(f'No parametric data found in file: {filepath}')
        
    # Count number of parts, use as spacer to jump to Tab 10, get parametric data
    elif parametric and parametric_possible:
        skipped_rows = len(parts_idx) + 10

        first_col = pd.read_csv(filepath,
                            usecols=[1],
                            names=['Part ID'],
                            skiprows=skipped_rows,
                            on_bad_lines='skip',
                            skip_blank_lines=True
                            )

        full_parts_list = []
        for value in first_col['Part ID']:
            if pd.isna(value):
                break
            if isinstance(value, str) and value.startswith('Tab - '):
                break
            if isinstance(value, str):
                base_value = value.replace('.1', '').replace('.s', '')
                if base_value in parts_data['Part ID'].values:
                    full_parts_list.append(value)

        if 'Tab - 10' and 'Tab - 11' in first_col['Part ID'].values:
            skipped_rows = 9 * (len(full_parts_list) + 4) + (len(parts_idx) + 10) 

        parts_params = pd.read_csv(filepath,
                            usecols=[7,9,10],
                            names=['Power','Point Distance','Exposure Time'],
                            skiprows=skipped_rows,
                            nrows=(len(parts_idx)),
                            on_bad_lines='skip',
                            skip_blank_lines=True
                            )
        
        parts_data = pd.concat([parts_data,parts_params],axis=1)
        return parts_data
    
    else:
        return parts_data


if __name__ == "__main__":
    untouched_file = Path.cwd() / Path('JR265_AMPM') / 'JR265_AMPM_parameters_all(fresh).csv'
    excel_file = Path.cwd() / 'JR265_parameters.csv'
    tab_file = Path.cwd() / Path('JR265_AMPM') / 'JR265_AMPM_parameters_volume(fresh).csv'
    fail_file = Path.cwd() / Path('JR265_AMPM') / 'JR265_AMPM_parameters_general(fresh).csv'
    
    
    df = get_parts(untouched_file,parametric=True)
    print(df)
