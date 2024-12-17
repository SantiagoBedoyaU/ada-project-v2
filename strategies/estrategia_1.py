from fastapi import UploadFile
from pyemd import emd
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import string
import time

def load_tpm(filename_tpm: str, num_elements: int):
    states = pd.Index(
        [np.binary_repr(i, width=num_elements)[::-1] for i in range(2**num_elements)]
    )
    tpm = np.loadtxt(filename_tpm, delimiter=",")
    df_tpm = pd.DataFrame(tpm, index=states, columns=states)
    return df_tpm, states

def load_tpm_2(filename_tpm: str, num_elements: int):
    states = pd.Index(
        [np.binary_repr(i, width=num_elements)[::-1] for i in range(2**num_elements)]
    )
    tpm = np.loadtxt(filename_tpm, delimiter=",")
    columns = pd.Index([np.binary_repr(i, width=1)[::-1] for i in range(2)])
    df_tpm = pd.DataFrame(tpm, index=states, columns=columns)
    return df_tpm, states

def load_tpm_3(filename_tpm: str, num_elements: int):
    states = pd.Index(
        [np.binary_repr(i, width=num_elements)[::-1] for i in range(2**num_elements)]
    )
    tpm = np.loadtxt(filename_tpm, delimiter=",")
    columns = pd.Index([np.binary_repr(i, width=num_elements)[::-1] for i in range(15)])
    df_tpm = pd.DataFrame(tpm, index=states, columns=columns)
    return df_tpm, states

def apply_background(df_tpm: pd.DataFrame, initial_state, candidate_system):
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
    result_df = df_tpm.loc[filtered_states, :]
    result_df = marginalize_cols(result_df.copy(), candidate_system)

    cut_filtered_states = []
    for state in result_df.index:
        state_list = list(state)
        for i in sorted(background_condition.keys(), reverse=True):
            state_list.pop(i)
        cut_filtered_states.append("".join(state_list))

    result_df.index = pd.Index(cut_filtered_states)
    result_df.columns = pd.Index(cut_filtered_states)
    return result_df

def marginalize_rows(df_tpm: pd.DataFrame, str_node_state: str, present_subsystem: str):
    key = str_node_state + present_subsystem
    if key in marginalized_tpm:
        return marginalized_tpm[key]
    else:
        df_tpm = df_tpm.sort_index()

        n_bits = len(df_tpm.index[0])
        if len(present_subsystem) != n_bits:
            raise ValueError("invalid present subsystem")

        positions_to_keep = [i for i, bit in enumerate(present_subsystem) if bit == "1"]

        def extract_bits(binary_str, positions):
            return "".join([binary_str[i] for i in positions])

        new_index = df_tpm.index.map(lambda x: extract_bits(x, positions_to_keep))
        result_df = df_tpm.groupby(new_index).mean()
        result_df = reorder_little_endian(result_df)
        marginalized_tpm[key] = result_df
        return result_df

def reorder_little_endian(df: pd.DataFrame):
    def bin_to_little_endian(bin_str):
        if not bin_str or not isinstance(bin_str, str):
            return 0
        bin_str = bin_str.strip()
        if not all(c in "01" for c in bin_str):
            return 0
        return int(bin_str[::-1], 2)  # Invertir string y convertir a entero base 2

    if df.empty:
        return df
    row_map = {idx: bin_to_little_endian(str(idx)) for idx in df.index}
    new_row_order = pd.Series(row_map).sort_values()
    col_map = {col: bin_to_little_endian(str(col)) for col in df.columns}
    new_col_order = pd.Series(col_map).sort_values()
    return df.reindex(index=new_row_order.index, columns=new_col_order.index)

def marginalize_cols(df_tpm: pd.DataFrame, future_subsystem: str):
    df_tpm = df_tpm.reindex(sorted(df_tpm.columns), axis=1)

    n_bits = len(df_tpm.columns[0])
    if len(future_subsystem) != n_bits:
        raise ValueError("invalid future subsystem")

    positions_to_keep = [i for i, bit in enumerate(future_subsystem) if bit == "1"]

    def extract_bits(binary_str, positions):
        return "".join([binary_str[i] for i in positions])
    
    new_index = df_tpm.columns.map(lambda x: extract_bits(x, positions_to_keep))
    result_df = df_tpm.T.groupby(new_index).sum()
    r = reorder_little_endian(result_df.T)
    return r

def tensor_product(df1: pd.DataFrame, df2: pd.DataFrame, keys_df1: list, keys_df2: list): 
    temp_data = {}
    if df1.index.tolist()[0] == df2.index.tolist()[0]:
        initial_state_label = df1.index.tolist()
    else:
        initial_state_label = 'I'    
    if len(keys_df1) == 0:
        df2.index = [initial_state_label[0]]
        return df2
    
    if len(keys_df2) == 0:
        df1.index = [initial_state_label[0]]
        return df1

    max_letter = max(keys_df1 + keys_df2)
    len_list = ord(max_letter) - ord('a') + 1

    labels = [' ' for _ in range(len_list)]

    for letter in keys_df1:
        idx = ord(letter) - ord('a')  # Convertimos la letra en índice
        labels[idx] = 2
    for letter in keys_df2:
        idx = ord(letter) - ord('a')  # Convertimos la letra en índice
        labels[idx] = 3
    
    filtered_labels = [item for item in labels if item != ' ']
    
    for df1_idx, df1_vals in df1.items():
        val_df1 = df1_vals.values.tolist()[0]
        for df2_idx, df2_vals in df2.items():
            val_df2 = df2_vals.tolist()[0]
            
            labels_copy = filtered_labels.copy()
            str_df1_idx = list(str(df1_idx))
            str_df2_idx = list(str(df2_idx))
            
            idx_digit = 0
            idx_digit_2 = 0

            for i in range(len(labels_copy)):
                if labels_copy[i] == 2 and idx_digit < len(str_df1_idx):
                    labels_copy[i] = int(str_df1_idx[idx_digit])
                    idx_digit += 1
                elif labels_copy[i] == 3 and idx_digit_2 < len(str_df2_idx):
                    labels_copy[i] = int(str_df2_idx[idx_digit_2])
                    idx_digit_2 += 1
            result = val_df1 * val_df2
            row_key = initial_state_label[0]
            col_key = "".join(map(str, labels_copy))
            temp_data.setdefault(row_key, {})[col_key] = result
    
    df_result = pd.DataFrame.from_dict(temp_data, orient="index").fillna(0)
    df_result = reorder_little_endian(df_result)
    return df_result

def tensor_product_of_matrix(df1: pd.DataFrame, df2: pd.DataFrame):
    result_dict = {}
    for df2col in df2.columns:
        for df1col in df1.columns:
            name = f"{df1col}{df2col}"
            result_dict[name] = df1[df1col] * df2[df2col]

    result = pd.DataFrame(result_dict)
    return result

def EMD(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    """
    Calculate the Earth Mover's Distance (EMD) between two probability
    distributions u and v.
    The Hamming distance was used as the ground metric.
    """
    if not all(isinstance(arr, np.ndarray) for arr in [u, v]):
        raise TypeError("u and v must be numpy arrays.")
    n: int = len(u)
    costs: NDArray[np.float64] = np.empty((n, n))
    for i in range(n):
        costs[i, :i] = [hamming_distance(i, j) for j in range(i)]
        costs[:i, i] = costs[i, :i]
    np.fill_diagonal(costs, 0)
    cost_matrix: NDArray[np.float64] = np.array(costs, dtype=np.float64)
    return emd(u, v, cost_matrix)

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def marginalize_node_states(
    df_tpm: pd.DataFrame,
    present: str,
    future: str,
    node_state: dict,
    initial_state: str,
    set_m: list
):
    results_node_states = {}
    present_idx = {idx: bit for idx, bit in enumerate(present) if bit == "1"}
    sorted_idx = sorted(present_idx.keys())
    label = ""
    for idx in sorted_idx:
        label += initial_state[idx]
    
    for elem in set_m:
        a = node_state[elem]     
        result_a = marginalize_rows(a.copy(), elem, present)

        if len(label) > 0:
            result_a = result_a.loc[[label], :]
        results_node_states[elem] = result_a

    keys = sorted(results_node_states.keys())
    if len(keys) > 0:
        first = keys[0]
    for i in range(1, len(keys)):
        results_node_states[keys[i]] = tensor_product(results_node_states[keys[i-1]], results_node_states[keys[i]], list(first), list(keys[i])) 
        first = "".join([first, keys[i]])

    if len(results_node_states) > 0:
        marginalizacion = results_node_states[keys[-1]]
    else:
        marginalizacion = marginalize_cols(df_tpm, future)
        marginalizacion = marginalize_rows(marginalizacion, '0', present)
        if len(label) > 0:
            marginalizacion = marginalizacion.loc[[label], :]
    return marginalizacion

def marginalize_node_states_1(
    df_tpm: pd.DataFrame,
    present: str,
    future: str,
    node_state: dict,
    set_m: list
):
    results_node_states = {}
    for elem in set_m:
        a = node_state[elem]     
        result_a = marginalize_rows(a.copy(), elem, present)
        results_node_states[elem] = result_a  
    
    keys = sorted(results_node_states.keys())
    if len(keys) > 0:
        first = keys[0]
    for i in range(1, len(keys)):
        results_node_states[keys[i]] = tensor_product_of_matrix(results_node_states[keys[i-1]], results_node_states[keys[i]]) 
        first = "".join([first, keys[i]])
        

    if len(results_node_states) > 0:
        marginalizacion = results_node_states[keys[-1]]
    else:
        marginalizacion = marginalize_cols(df_tpm, future)
        marginalizacion = marginalize_rows(marginalizacion, '0', present)
    marginalized_tpm.clear()
    return marginalizacion

def bipartition_system(
    df_tpm: pd.DataFrame,
    v: list,
    initial_state: str,
    candidates_bipartition: list,
    node_state: dict,
):
    if len(v) <= 2:
        candidates_bipartition.append(v[-1])
        return candidates_bipartition
    w_1 = [v[0]]
    w_1l = []
    wp = [item for item in v if item not in w_1]

    results_u = {}
    
    present_v, _, _ = set_to_binary(global_v, v)
    present_idx = {idx: bit for idx, bit in enumerate(present_v) if bit == "1"}
    sorted_idx = sorted(present_idx.keys())
    label = ""
    for idx in sorted_idx:
        label += initial_state[idx]
    
    initial_state_values = df_tpm.loc[label, :].values
    
    while len(wp) > 0:
        for u in wp:
            w_1u = w_1.copy()
            w_1u.append(u)
            w_1up = [item for item in v if item not in w_1u]
            
            #----------------MARGINALIZACION W_1U ----------------
            present, future, set_m_w1 = set_to_binary(global_v, w_1u)
            marginalizacionW_1u = marginalize_node_states(
                df_tpm, present, future, node_state, initial_state, set_m_w1
            )
            #----------------MARGINALIZACION W_1UP--------------
            present, future, set_m_wp = set_to_binary(global_v, w_1up)
            marginalizacionW_1up = marginalize_node_states(
                df_tpm, present, future, node_state, initial_state, set_m_wp
            )
            #----------------TENSOR_PRODUCT---------------------
            first_product_result = tensor_product(
                marginalizacionW_1u, marginalizacionW_1up, set_m_w1, set_m_wp
            )
            #----------------FIRST_EMD---------------------
            first_product_result = np.array(first_product_result).flatten().astype(np.float64)
            initial_state_values = np.array(initial_state_values).astype(np.float64)   
            emd1 = EMD(first_product_result, initial_state_values)
            #------------- MARGINALIZACION U ---------------
            up = [item for item in v if item not in [u]]
            present, future, set_mu = set_to_binary(global_v, [u])
            
            marginalizacionU = marginalize_node_states(
                df_tpm, present, future, node_state, initial_state, set_mu
            )
            #----------------MARGINALIZACIÓN UP ----------------
            present, future, set_mp = set_to_binary(global_v, up)
            marginalizacionUp = marginalize_node_states(
                df_tpm, present, future, node_state, initial_state, set_mp
            )
            #----------------TENSOR_PRODUCT---------------------
            second_product_result = tensor_product(
                marginalizacionU, marginalizacionUp, set_mu, set_mp
            )
            #----------------SECOND_EMD---------------------
            second_product_result = np.array(second_product_result).flatten().astype(np.float64)
            emd2 = EMD(second_product_result, initial_state_values)
            result_emd = emd1 - emd2

            if isinstance(u, list):
                results_u[tuple(u)] = result_emd
            else:
                results_u[u] = result_emd

        #-------------------RESULTS--------------------
        min_result = min(results_u.values())
        key = [key for key, value in results_u.items() if value == min_result][0]
        if isinstance(key, tuple):
            key = list(key)
        wp.remove(key)
        w_1.append(key)
        w_1l.append(key)
        results_u.clear()

    candidates_bipartition.append(w_1l[-1])
    v.remove(w_1l[-1])
    v.remove(w_1l[-2])
    if isinstance(w_1l[-1], list) and isinstance(w_1l[-2], list):
        v.append(w_1l[-1] + w_1l[-2])

    elif isinstance(w_1l[-2], list):
        v.append([w_1l[-1]] + w_1l[-2])

    elif isinstance(w_1l[-1], list):
        v.append(w_1l[-1] + [w_1l[-2]])
    else:
        v.append([w_1l[-1], w_1l[-2]])
    candidates_bipartition = bipartition_system(
        df_tpm, v, initial_state, candidates_bipartition, node_state
    )
    return candidates_bipartition

def set_to_binary_1(set: list, present_label_len: int, future_label_len: int):
    abc = string.ascii_lowercase
    binary_present = list(np.binary_repr(0, present_label_len))
    binary_future = list(np.binary_repr(0, future_label_len))
    for elem in set:
        if isinstance(elem, list):
            for elem_2 in elem:
                idx = abc.index(elem_2[0])
                if "t+1" in elem_2:
                    binary_future[idx] = "1"
                else:
                    binary_present[idx] = "1"
        else:
            idx = abc.index(elem[0])
            if "t+1" in elem:
                binary_future[idx] = "1"
            else:
                binary_present[idx] = "1"

    return ["".join(binary_present), "".join(binary_future)]

def set_to_binary(global_v: list, subset: list):
    positions_to_keep_present = []
    positions_to_keep_future = []
    group_t1_in_subset = []
    group_t1_in_subset_letters = []
    
    group_t = [elemento for elemento in global_v if "t" in elemento and "t+1" not in elemento]
    group_t1 = [elemento for elemento in global_v if "t+1" in elemento]
    
    group_t_letters = [elem[0] for elem in group_t]
    indice_t = ord(max(group_t_letters)) - ord('a')
    
    group_t_1_letters = [elem[0] for elem in group_t1]
    indice_t_1 = ord(max(group_t_1_letters)) - ord('a')
    
    for item in subset:
        if isinstance(item, list):
            for it in item:
                if it in group_t1:
                    group_t1_in_subset.append(it)
        else:
            if item in group_t1:
                group_t1_in_subset.append(item)
    
    if len(group_t1_in_subset) > 0:
        group_t1_in_subset_letters = [elem[0] for elem in group_t1_in_subset]
    
    binary_present = list(np.binary_repr(0, indice_t + 1))
    binary_future = list(np.binary_repr(0, indice_t_1 + 1))
    
    abc = string.ascii_lowercase

    for i in range(len(binary_future)):
        binary_future[i] = '0'
        
    for j in range(len(binary_present)):
        binary_present[j] = '0'
    
    for elem in subset:
        if isinstance(elem, list):
            for elem_2 in elem:
                idx = abc.index(elem_2[0])
                if "t+1" in elem_2:
                    binary_future[idx] = "1"
                else:
                    binary_present[idx] = "1"
        else:
            idx = abc.index(elem[0])
            if "t+1" in elem:
                binary_future[idx] = "1"
            else:
                binary_present[idx] = "1"
                
    for k in group_t_letters:
        positions_to_keep_present.append(ord(k) - ord('a'))
        
    for l in group_t_1_letters:
        positions_to_keep_future.append(ord(l) - ord('a'))
        
                
    binary_future = [binary_future[i] for i in positions_to_keep_future]
    binary_present = [binary_present[i] for i in positions_to_keep_present]
                
    return ["".join(binary_present), "".join(binary_future), group_t1_in_subset_letters]

def get_matrices_node_state(df_tpm: pd.DataFrame, future_subsystem: list):
    matrices_node_state = {}
    abc = string.ascii_lowercase
    cols = "0" * len(df_tpm.columns[0])
    k = 0
    for i in range(len(future_subsystem)):
        if future_subsystem[i] == '1':
            future = cols[:k] + "1" + cols[k + 1 :]
            matrix = marginalize_cols(df_tpm.copy(), future)
            matrices_node_state[abc[i]] = matrix
            k += 1
    return matrices_node_state
     
def get_first_matrices_node_state(df_tpm: pd.DataFrame):
    matrices_node_state = {}
    abc = string.ascii_lowercase
    cols = "0" * len(df_tpm.columns[0])
    k = 0
    for i in range(len(df_tpm.columns[0])):
        future = cols[:k] + "1" + cols[k + 1 :]
        matrix = marginalize_cols(df_tpm.copy(), future)
        matrices_node_state[abc[i]] = matrix
        k += 1
    return matrices_node_state

def build_v(present_subsystem: str, future_subsystem: str):
    v = []
    abc = string.ascii_lowercase
    for idx, bit in enumerate(present_subsystem):
        if bit == "1":
            v.append(f"{abc[idx]}_t")
    for idx, bit in enumerate(future_subsystem):
        if bit == "1":
            v.append(f"{abc[idx]}_t+1")
    return v

def min_EMD(
    df_tpm: pd.DataFrame, v: list[str], bipartion_list: list[str], label: str, node_states, initial_state
):
    initial_state_values = df_tpm.loc[label, :].values
    initial_state_values = np.array(initial_state_values).astype(np.float64)
    
    emd_results = {}
    for elem in bipartion_list:
        elemP = []
        if isinstance(elem, list):
            for e in v:
                if e not in elem:
                    elemP.append(e)
        else:
            elemP = [e for e in v if e != elem]
        #-------------------------- Marginalización Elem ---------------------------
        presentElem, futureElem, set_m_elem = set_to_binary(global_v, [elem])
        marginalizacionElem = marginalize_node_states(df_tpm, presentElem, futureElem, node_states, initial_state, set_m_elem)
        #-------------------------- Marginalización Elemp --------------------------
        presentElemP, futureElemP, set_m_elemp = set_to_binary(global_v, [elemP])
        marginalizacionElemP = marginalize_node_states(df_tpm, presentElemP, futureElemP, node_states, initial_state, set_m_elemp)
        #--------------------- TENSOR_PRODUCT --------------------------
        tensor_result = tensor_product(
            marginalizacionElem, marginalizacionElemP, set_m_elem, set_m_elemp
        )
        tensor_result_2 = np.array(tensor_result).flatten().astype(np.float64)
        #--------------------- EMD --------------------------
        emd = EMD(tensor_result_2, initial_state_values)
        emd_results[tuple(elem)] = emd
    min_emd_result = min(emd_results.values())
    min_emd_key = [
        key for key, value in emd_results.items() if value == min_emd_result
    ][0]
    return [min_emd_key, min_emd_result]

# caso de prueba red 10
def main():
    inicio = time.perf_counter()
    [
        initial_state_str,
        candidate_system_str,
        present_subsystem_str,
        future_subsystem_str,
    ] = np.loadtxt("./red10/system_values.csv", delimiter=",", skiprows=1, dtype=str)
    initial_state = initial_state_str.strip()
    candidate_system = candidate_system_str.strip()
    present_subsystem = present_subsystem_str.strip()
    future_subsystem = future_subsystem_str.strip()
    print(f"{initial_state=}, {candidate_system=}, {present_subsystem=}, {future_subsystem=}")
    matrix_1, _ = load_tpm_2("./red10/state_node_a.csv", len(candidate_system))
    matrix_2, _ = load_tpm_2("./red10/state_node_b.csv", len(candidate_system))
    matrix_3, _ = load_tpm_2("./red10/state_node_c.csv", len(candidate_system))
    matrix_4, _ = load_tpm_2("./red10/state_node_d.csv", len(candidate_system))
    matrix_5, _ = load_tpm_2("./red10/state_node_e.csv", len(candidate_system))
    matrix_6, _ = load_tpm_2("./red10/state_node_f.csv", len(candidate_system))
    matrix_7, _ = load_tpm_2("./red10/state_node_g.csv", len(candidate_system))
    matrix_8, _ = load_tpm_2("./red10/state_node_h.csv", len(candidate_system))
    matrix_9, _ = load_tpm_2("./red10/state_node_i.csv", len(candidate_system))
    matrix_10, _ = load_tpm_2("./red10/state_node_j.csv", len(candidate_system))
    
    tensor_flow = tensor_product_of_matrix(matrix_1, matrix_2)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_3)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_4)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_5)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_6)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_7)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_8)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_9)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_10)

    df_tpm = apply_background(tensor_flow, initial_state, candidate_system)
    
    v = build_v(present_subsystem, future_subsystem)

    global global_v 
    global_v = v.copy()
    global marginalized_tpm
    marginalized_tpm = {}
    
    present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))

    node_states = get_first_matrices_node_state(df_tpm)

    result_df = marginalize_node_states_1(df_tpm, present, future, node_states, sorted(node_states.keys()))
    result_df = marginalize_cols(result_df, future)
    node_states = get_matrices_node_state(result_df, future)
    
    candidates_bipartition = []
    candidate_bipartitions = bipartition_system(
        result_df.copy(), v.copy(), initial_state, candidates_bipartition, node_states
    )
    initial_state_v, _, _ = set_to_binary(global_v, v)
    present_idx = {idx: bit for idx, bit in enumerate(initial_state_v) if bit == "1"}
    sorted_idx = sorted(present_idx.keys())
    label = ""
    for idx in sorted_idx:
        label += initial_state[idx]
        
    [min_emd_key, min_emd_result] = min_EMD(
        result_df.copy(), v.copy(), candidate_bipartitions, label, node_states, initial_state
    )
    print(f"{min_emd_key=}, {min_emd_result=}")
    fin = time.perf_counter()
    print("Tiempo=")
    print(fin-inicio)

# casos de prueba red 5
def main_2():
    inicio = time.perf_counter()
    [
        initial_state_str,
        candidate_system_str,
        present_subsystem_str,
        future_subsystem_str,
    ] = np.loadtxt("./red5/system_values.csv", delimiter=",", skiprows=1, dtype=str)
    initial_state = initial_state_str.strip()
    candidate_system = candidate_system_str.strip()
    present_subsystem = present_subsystem_str.strip()
    future_subsystem = future_subsystem_str.strip()
    matrix_1, _ = load_tpm_2("./red5/state_node_a.csv", len(candidate_system))
    matrix_2, _ = load_tpm_2("./red5/state_node_b.csv", len(candidate_system))
    matrix_3, _ = load_tpm_2("./red5/state_node_c.csv", len(candidate_system))
    matrix_4, _ = load_tpm_2("./red5/state_node_d.csv", len(candidate_system))
    matrix_5, _ = load_tpm_2("./red5/state_node_e.csv", len(candidate_system))
    
    tensor_flow = tensor_product_of_matrix(matrix_1, matrix_2)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_3)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_4)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_5)

    print(f"{initial_state=}, {candidate_system=}, {present_subsystem=}, {future_subsystem=}")
    df_tpm = apply_background(tensor_flow, initial_state, candidate_system)
    
    v = build_v(present_subsystem, future_subsystem)

    global global_v 
    global_v = v.copy()
    global marginalized_tpm
    marginalized_tpm = {}
    
    present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))

    node_states = get_first_matrices_node_state(df_tpm)

    result_df = marginalize_node_states_1(df_tpm, present, future, node_states, sorted(node_states.keys()))
    result_df = marginalize_cols(result_df, future)
    node_states = get_matrices_node_state(result_df, future)
    
    candidates_bipartition = []
    candidate_bipartitions = bipartition_system(
        result_df.copy(), v.copy(), initial_state, candidates_bipartition, node_states
    )
    initial_state_v, _, _ = set_to_binary(global_v, v)
    present_idx = {idx: bit for idx, bit in enumerate(initial_state_v) if bit == "1"}
    sorted_idx = sorted(present_idx.keys())
    label = ""
    for idx in sorted_idx:
        label += initial_state[idx]
        
    [min_emd_key, min_emd_result] = min_EMD(
        result_df.copy(), v.copy(), candidate_bipartitions, label, node_states, initial_state
    )
    print(f"{min_emd_key=}, {min_emd_result=}")
    fin = time.perf_counter()
    print("Tiempo=")
    print(fin-inicio)

##############################################
# Caso de prueba red 15
##############################################
def main_3():
    inicio = time.perf_counter()
    [
        initial_state_str,
        candidate_system_str,
        present_subsystem_str,
        future_subsystem_str,
    ] = np.loadtxt("system_values_3.csv", delimiter=",", skiprows=1, dtype=str)
    initial_state = initial_state_str.strip()
    candidate_system = candidate_system_str.strip()
    present_subsystem = present_subsystem_str.strip()
    future_subsystem = future_subsystem_str.strip()
    matrix, states = load_tpm("resultado_15.csv", len(candidate_system))


    # print(tensor_flow)
    print(f"{initial_state=}, {candidate_system=}, {present_subsystem=}, {future_subsystem=}")
    df_tpm = apply_background(matrix, initial_state, candidate_system)
    # print(df_tpm)

    v = build_v(present_subsystem, future_subsystem)

    global global_v 

    global_v = v.copy()

    present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))
    result_df = marginalize_cols(df_tpm, future)
    result_df = marginalize_rows(result_df, present)

    node_states = get_matrices_node_state(result_df)


    candidates_bipartition = []
    candidate_bipartitions = bipartition_system(
        result_df.copy(), v.copy(), initial_state, candidates_bipartition, node_states
    )
    print(f"{candidate_bipartitions=}")
    initial_state_v, _ = set_to_binary(global_v, v)
    present_idx = {idx: bit for idx, bit in enumerate(initial_state_v) if bit == "1"}
    sorted_idx = sorted(present_idx.keys())
    label = ""
    for idx in sorted_idx:
        label += initial_state[idx]
        
    [min_emd_key, min_emd_result] = min_EMD(
        result_df.copy(), v.copy(), candidate_bipartitions, label
    )
    print(f"{min_emd_key=}, {min_emd_result=}")
    fin = time.perf_counter()
    print("Tiempo=")
    print(fin-inicio)

async def solve(
    tpms: list[UploadFile], 
    initial_state: str, 
    candidate_system: str, 
    present_subsystem: str,
    future_subsystem: str,
):
    print(f"{initial_state=}, {candidate_system=}, {present_subsystem=}, {future_subsystem=}")
    n_elements = len(candidate_system)
    matrix1, _ = load_tpm_2(tpms[0].file, n_elements)
    matrix2, _ = load_tpm_2(tpms[1].file, n_elements)
    tensor_product = tensor_product_of_matrix(matrix1, matrix2)
    for i in range(2, len(tpms)):
        matrix, _ = load_tpm_2(tpms[i].file, n_elements)
        tensor_product = tensor_product_of_matrix(tensor_product, matrix)

    df_tpm = apply_background(tensor_product, initial_state, candidate_system)

    v = build_v(present_subsystem, future_subsystem)

    global global_v 
    global_v = v.copy()
    global marginalized_tpm
    marginalized_tpm = {}

    present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))

    node_states = get_first_matrices_node_state(df_tpm)

    result_df = marginalize_node_states_1(df_tpm, present, future, node_states, sorted(node_states.keys()))
    result_df = marginalize_cols(result_df, future)
    node_states = get_matrices_node_state(result_df, future)

    candidates_bipartition = []
    inicio = time.perf_counter()
    candidate_bipartitions = bipartition_system(
        result_df.copy(), v.copy(), initial_state, candidates_bipartition, node_states
    )
    fin = time.perf_counter()
    initial_state_v, _, _ = set_to_binary(global_v, v)
    present_idx = {idx: bit for idx, bit in enumerate(initial_state_v) if bit == "1"}
    sorted_idx = sorted(present_idx.keys())
    label = ""
    for idx in sorted_idx:
        label += initial_state[idx]
        
    [min_emd_key, min_emd_result] = min_EMD(
        result_df.copy(), v.copy(), candidate_bipartitions, label, node_states, initial_state
    )
    return [min_emd_key, min_emd_result, fin-inicio]
    
# main()
