import pandas as pd
from pathlib import Path


file1 = Path('/Users/guggoo/Documents/Programming/Matlab/JR265_parameters.csv')
file2 = Path('/Users/guggoo/Documents/Programming/Matlab/JR291_parameters_all.csv')
file3 = Path('/Users/guggoo/Documents/Programming/Matlab/JR_291_parameters_tab1.csv')


def get_parts(filepath : str | Path) -> pd.DataFrame:
    """
    Create pd.DataFrame from part data exported from QuantAM (.csv)

    Parameters:
    -----------
    filepath : str or Path
        Filepath to part data file (.csv)

    Returns
    -----------
    part_data : pd.DataFrame
        DataFrame of Part ID, Layer Thickness, X Position, Y Position, Layers Count
    """

    filepath = Path(filepath) if type(filepath) == str else filepath
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    part_data = pd.read_csv(filepath,
                    usecols=[1,3,4,5,6],
                    names=['Part ID','Layer Thickness','X Position','Y Position','Layers Count'],
                    skiprows=6,
                    on_bad_lines='skip',
                    skip_blank_lines=True
                    )

    part_nums = []
    for value in part_data['Part ID']:
        if type(value) == float:
            break
        if value == 'Tab - 1':
            break
        else:
            part_nums.append(int(value)-1)

    part_data = part_data.loc[part_nums]
    
    return part_data


df = get_parts(file1)
print(df)
