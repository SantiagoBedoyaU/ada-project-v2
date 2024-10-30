import numpy as np
import pandas as pd


def load_tpm(filename_tpm: str, num_elements: int):
    states = pd.Index(
        [np.binary_repr(i, width=num_elements)[::-1] for i in range(2**num_elements)]
    )
    tpm = np.loadtxt(filename_tpm, delimiter=",")
    df_tpm = pd.DataFrame(tpm, index=states, columns=states)

    return df_tpm, states


def apply_background(df_tpm, initial_state, candidate_system):
    background_condition = {
        idx: initial_state[idx] 
        for idx, bit in enumerate(candidate_system) 
        if bit == "0"
    }
    filtered_states = [
        state
        for state in df_tpm.index
        if all(state[i] == bit for i, bit in background_condition.items())
    ]
    result_df = df_tpm.loc[filtered_states, filtered_states]
    return result_df


def marginalize_rows(df_tpm, present_subsystem: str):
    n_bits = len(df_tpm.index[0])
    if len(present_subsystem) != n_bits:
        raise ValueError("invalid present subsystem")

    positions_to_keep = [i for i, bit in enumerate(present_subsystem) if bit == '1']

    def extract_bits(binary_str, positions):
        return ''.join([binary_str[i] for i in positions])

    new_index = df_tpm.index.map(lambda x: extract_bits(x, positions_to_keep))

    result_df = df_tpm.groupby(new_index).mean()
    return result_df

def marginalize_cols(df_tpm, future_subsystem: str):
    n_bits = len(df_tpm.columns[0])
    if len(future_subsystem) != n_bits:
        raise ValueError("invalid future subsystem")

    positions_to_keep = [i for i, bit in enumerate(future_subsystem) if bit == '1']

    def extract_bits(binary_str, positions):
        return ''.join([binary_str[i] for i in positions])

    new_index = df_tpm.columns.map(lambda x: extract_bits(x, positions_to_keep))

    result_df = df_tpm.T.groupby(new_index).sum()
    return result_df.T


def apply_marginalize(df_tpm, elements):
    df_tpm_marginalize_rows = marginalize_rows(df_tpm, elements)
    df_marginalized = marginalize_cols(df_tpm_marginalize_rows, elements)
    return df_marginalized


[initial_state_str, candidate_system_str, present_subsystem_str, future_subsystem_str] = np.loadtxt(
    "system_values.csv", delimiter=",", skiprows=1, dtype=str
)
initial_state = initial_state_str.strip()
candidate_system = candidate_system_str.strip()
present_subsystem = present_subsystem_str.strip()
future_subsystem = future_subsystem_str.strip()

df_tpm, states = load_tpm("matrix_guia.csv", len(candidate_system))
# print(df_tpm)

result_df = apply_background(df_tpm, initial_state, candidate_system)
# print(result_df)

result_df = marginalize_rows(result_df, present_subsystem)
# print(result_df)

result_df = marginalize_cols(result_df, future_subsystem)
print(result_df)