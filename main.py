import numpy as np
import pandas as pd


def load_tpm(filename_tpm: str, num_elements: int):
    states = pd.Index(
        [np.binary_repr(i, width=num_elements)[::-1] for i in range(2**num_elements)]
    )
    tpm = np.loadtxt(filename_tpm, delimiter=",")
    df_tpm = pd.DataFrame(tpm, index=states, columns=states)

    return df_tpm, states


def apply_background(df_tpm, candidate_system):
    background_condition = {
        idx: "0" for idx, bit in enumerate(candidate_system) if bit == "0"
    }
    filtered_states = [
        state
        for state in df_tpm.index
        if all(state[i] == bit for i, bit in background_condition.items())
    ]
    result_df = df_tpm.loc[filtered_states, filtered_states]
    return result_df


def marginalize_rows(df_tpm, var_to_marginalize):
    var_positions = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    all_vars = set('ABCD')
    vars_to_keep = all_vars - set(var_to_marginalize)
    positions_to_keep = [var_positions[var] for var in vars_to_keep]

    def extract_bits(binary_str, positions):
        return ''.join([binary_str[i] for i in positions])

    new_index = df_tpm.index.map(lambda x: extract_bits(x, positions_to_keep))
    result_df = df_tpm.groupby(new_index).mean()

    return result_df

def marginalize_cols(df_tpm, elements):
    # filterd_states = df_tpm.colums
    for elem in elements:
        df_tpm = df_tpm.groupby(
            lambda state: state[:elem] + state[elem + 1 :], axis=1
        ).sum()
    return df_tpm


def apply_marginalize(df_tpm, elements):
    df_tpm_marginalize_rows = marginalize_rows(df_tpm, elements)
    df_marginalized = marginalize_cols(df_tpm_marginalize_rows, elements)
    return df_marginalized


[initial_state_str, candidate_system_str] = np.loadtxt(
    "system_values.csv", delimiter=",", skiprows=1, dtype=str
)
initial_state = initial_state_str.strip()
candidate_system = candidate_system_str.strip()

df_tpm, states = load_tpm("matrix_guia.csv", len(candidate_system))
# print(df_tpm)

result_df = apply_background(df_tpm, candidate_system)
print(result_df)
print()

result_df = marginalize_rows(result_df, 'BCD')
print(result_df)
# result_df = marginalize_rows(result_df, [1, 2, 3])
# print(result_df)
# result_df = marginalize_rows(result_df, [])
