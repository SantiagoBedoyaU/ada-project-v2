import numpy as np
import pandas as pd
from pyemd import emd
from numpy.typing import NDArray
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


def marginalize_rows(df_tpm: pd.DataFrame, present_subsystem: str):
    df_tpm = df_tpm.sort_index()

    n_bits = len(df_tpm.index[0])
    if len(present_subsystem) != n_bits:
        raise ValueError("invalid present subsystem")

    positions_to_keep = [i for i, bit in enumerate(present_subsystem) if bit == "1"]

    def extract_bits(binary_str, positions):
        return "".join([binary_str[i] for i in positions])

    new_index = df_tpm.index.map(lambda x: extract_bits(x, positions_to_keep))
    result_df = df_tpm.groupby(new_index).mean()
    return reorder_little_endian(result_df)


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


def tensor_product(df1: pd.DataFrame, df2: pd.DataFrame):
    """
               AC
               00   10   01   11
        df1 = [1.0, 0.0, 1.0, 0.0]

               BD
               00   10   01    11
        df2 = [1.0, 0.0, 1.0, 0.0]

    """

    # if type(df1[0]) is np.ndarray:
    #     df1 = df1[0]
    # if type(df2[0]) is np.ndarray:
    #     df2 = df2[0]
    print("++++++++++++++")
    print(df1)
    print()
    print(df2)
    print("++++++++++++++")

    df1_columns = ['000', '100', '010', '110', '001', '101', '011', '111']
    df2_columns = ['000', '100', '010', '110', '001', '101', '011', '111']

    def calc_key():
        pass

    labels: dict[str, float] = {} # labels.keys() = []
    for df1_idx, df1_vals in df1.items():
        column_df1 = df1_idx
        val_df1 = df1_vals.values.tolist()[0]
        for df2_idx, df2_vals in df2.items():
            column_df2 = df2_idx
            val_df2 = df2_vals.tolist()[0]

            result = val_df1 * val_df2
        print(f"{df1_idx=}, {df1_vals=}, {df1_vals.values.tolist()=}")
        input()

    result = []
    for df2_elem in df2:
        for df1_elem in df1:
            result.append(df1_elem * df2_elem)
    return result


def tensor_product_of_matrix(df1: pd.DataFrame, df2: pd.DataFrame):
    result = pd.DataFrame()
    for df2col in df2.columns:
        for df1col in df1.columns:
            name = f"{df1col}{df2col}"
            result[name] = df1[df1col] * df2[df2col]
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
):
    results_node_states = []
    present_idx = {idx: bit for idx, bit in enumerate(present) if bit == "1"}
    sorted_idx = sorted(present_idx.keys())
    label = ""
    for idx in sorted_idx:
        label += initial_state[idx]
    if "1" in future:
        for idx, bit in enumerate(future):
            if bit == "1":
                a = node_state[idx]
                
                result_a = marginalize_rows(a.copy(), present)

                if len(label) > 0:
                    result_a = result_a.loc[[label], :]
                
                print(f"{label=}")
                results_node_states.append(result_a)

        for i in range(1, len(results_node_states)):
            first = results_node_states[i - 1]
            second = results_node_states[i]

            results_node_states[i] = tensor_product_of_matrix(first, second)

        if len(results_node_states) > 0:
            marginalizacion = results_node_states[-1]
            print("<--------producto tensorial de las nodo_estado--------->")
            print(marginalizacion)

    else:
        marginalizacion = marginalize_cols(df_tpm, future)
        marginalizacion = marginalize_rows(marginalizacion, present)
        if len(label) > 0:
            marginalizacion = marginalizacion.loc[[label], :]
        print(marginalizacion)

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
    
    present_v, _ = set_to_binary(global_v, v)
    present_idx = {idx: bit for idx, bit in enumerate(present_v) if bit == "1"}
    sorted_idx = sorted(present_idx.keys())
    label = ""
    for idx in sorted_idx:
        label += initial_state[idx]
    
    print(f"{present_v=}")
    print(f"{label=}")
    
    initial_state_values = df_tpm.loc[label, :].values
    print(f"{initial_state_values=}")
    print()
    
    while len(wp) > 0:
        for u in wp:
            w_1u = w_1.copy()
            w_1u.append(u)
            w_1up = [item for item in v if item not in w_1u]
            
            print("----------------MARGINALIZACIÓN W_1U ----------------")
            print(f"{w_1u=}")
            print(f"{w_1up=}")
            print()
            
            # marginalization of W_1u
            present, future = set_to_binary(global_v, w_1u)
            
            marginalizacionW_1u = marginalize_node_states(
                df_tpm, present, future, node_state, initial_state
            )
            
            
            print("----------------MARGINALIZACIÓN W_1UP--------------")
            
            # marginalization of w_1up
            present, future = set_to_binary(global_v, w_1up)
            
            marginalizacionW_1up = marginalize_node_states(
                df_tpm, present, future, node_state, initial_state
            )
            
            # tensor_product
            first_product_result = tensor_product(
                marginalizacionW_1u, marginalizacionW_1up
            )
            
            print(f"{first_product_result=}")
            print()

            # 1 EMD
            first_product_result = np.array(first_product_result).astype(np.float64)
            initial_state_values = np.array(initial_state_values).astype(np.float64)
            
            emd1 = EMD(first_product_result, initial_state_values)
            
            print(f"{emd1=}")
            print()
            
            up = [item for item in v if item not in [u]]
            
            print("------------------------------ MARGINALIZACIÓN U ---------------------------------")
            
            present, future = set_to_binary(global_v, [u])
            
            marginalizacionU = marginalize_node_states(
                df_tpm, present, future, node_state, initial_state
            )

            print("----------------MARGINALIZACIÓN UP ----------------")

            present, future = set_to_binary(global_v, up)
            
            marginalizacionUp = marginalize_node_states(
                df_tpm, present, future, node_state, initial_state
            )
            

            second_product_result = tensor_product(
                marginalizacionU.values, marginalizacionUp.values
            )
            print(f"{second_product_result=}")
            print()
            

            second_product_result = np.array(second_product_result).astype(np.float64)
            emd2 = EMD(second_product_result, initial_state_values)
            
            print(f"{emd2=}")
            print()
            
            result_emd = emd1 - emd2

            if isinstance(u, list):
                results_u[tuple(u)] = result_emd
            else:
                results_u[u] = result_emd

        print("----------------------------------RESULTS-------------------------------------")
        print(f"{results_u=}")
        print()
        min_result = min(results_u.values())
        key = [key for key, value in results_u.items() if value == min_result][0]
        if isinstance(key, tuple):
            key = list(key)
        wp.remove(key)
        w_1.append(key)
        w_1l.append(key)
        
        print("Nuevo w_1:")
        print(f"{w_1=}")
        print()
        print("Nuevo wp:")
        print(f"{wp=}")
        print()
        

        results_u.clear()

    candidates_bipartition.append(w_1l[-1])
    print(f"{candidates_bipartition=}")
    print()
    v.remove(w_1l[-1])
    v.remove(w_1l[-2])
    print("Nuevo V, debería no tener dos elementos (candidatos)")
    print(f"{v=}")
    print()
    if isinstance(w_1l[-1], list) and isinstance(w_1l[-2], list):
        v.append(w_1l[-1] + w_1l[-2])

    elif isinstance(w_1l[-2], list):
        v.append([w_1l[-1]] + w_1l[-2])

    elif isinstance(w_1l[-1], list):
        v.append(w_1l[-1] + [w_1l[-2]])
    else:
        v.append([w_1l[-1], w_1l[-2]])

    print("Nuevo V, debería tener los elementos unidos (candidatos)")
    print(f"{v=}")
    print()
    
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

def set_to_binary(global_v: list, set: list):
    
    positions_to_keep_present = []
    positions_to_keep_future = []
    
    group_t = [elemento for elemento in global_v if "t" in elemento and "t+1" not in elemento]
    group_t1 = [elemento for elemento in global_v if "t+1" in elemento]
    
    group_t_letters = [elem[0] for elem in group_t]
    indice_t = ord(max(group_t_letters)) - ord('a')
    
    group_t_1_letters = [elem[0] for elem in group_t1]
    indice_t_1 = ord(max(group_t_1_letters)) - ord('a')
    
    binary_present = list(np.binary_repr(0, indice_t + 1))
    binary_future = list(np.binary_repr(0, indice_t_1 + 1))
    
    abc = string.ascii_lowercase

    for i in range(len(binary_future)):
        binary_future[i] = '0'
        
    for j in range(len(binary_present)):
        binary_present[j] = '0'
    
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
                
    for k in group_t_letters:
        positions_to_keep_present.append(ord(k) - ord('a'))
        
    for l in group_t_1_letters:
        positions_to_keep_future.append(ord(l) - ord('a'))
        
                
    binary_future = [binary_future[i] for i in positions_to_keep_future]
    binary_present = [binary_present[i] for i in positions_to_keep_present]
                
    return ["".join(binary_present), "".join(binary_future)]

def get_matrices_node_state(df_tpm: pd.DataFrame):
    matrices_node_state = {}
    string = "0" * len(df_tpm.columns[0])
    for i in range(len(df_tpm.columns[0])):
        future = string[:i] + "1" + string[i + 1 :]
        matrix = marginalize_cols(df_tpm.copy(), future)
        matrices_node_state[i] = matrix
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
    print(f"{initial_state_values=}")
    
    emd_results = {}
    for elem in bipartion_list:
        elemP = []
        if isinstance(elem, list):
            for e in v:
                if e not in elem:
                    elemP.append(e)
        else:
            elemP = [e for e in v if e != elem]
        
        
        print(f"{elem=}")
        print(f"{elemP=}")
        print()

        print("-------------------------- Marginalización Elem ---------------------------")
        
        presentElem, futureElem = set_to_binary(global_v, [elem])
        print(f"{presentElem=}")
        print(f"{futureElem=}")
        print()
        # present_idx = {idx: bit for idx, bit in enumerate(presentElem) if bit == "1"}
        # sorted_idx = sorted(present_idx.keys())
        # label = ""
        # for idx in sorted_idx:
        #     label += initial_state[idx]
            
        # print(f"{label=}")
        # print()
        
        marginalizacionElem = marginalize_node_states(df_tpm, presentElem, futureElem, node_states, initial_state)
        # marginalizacionElem = marginalize_rows(df_tpm, presentElem) # COMPROBAR SI NO NECESITA A LA PUTA_FUNCTION()     
        # marginalizacionElem = marginalize_cols(marginalizacionElem, futureElem)
        
        print(f"{marginalizacionElem=}")
        print()
        
        # if len(label) > 0:
        #     marginalizacionElem = marginalizacionElem.loc[label, :]
        
        print("-------------------------- Marginalización Elemp ---------------------------")

        presentElemP, futureElemP = set_to_binary(global_v, [elemP])
        print(f"{presentElemP=}")
        print(f"{futureElemP=}")
        print()
        # present_idx = {idx: bit for idx, bit in enumerate(presentElemP) if bit == "1"}
        # sorted_idx = sorted(present_idx.keys())
        # label = ""
        # for idx in sorted_idx:
        #     label += initial_state[idx]
            
        # print(f"{label=}")
        # print()
        marginalizacionElemP = marginalize_node_states(df_tpm, presentElemP, futureElemP, node_states, initial_state)
        
        print(f"{marginalizacionElemP=}")
        print()
        
        print(f"{label=}")
        
        # if len(label) > 0:
        #     marginalizacionElemP = marginalizacionElemP.loc[label, :]
            
        print("Después de ubicar la fila inicial")
        print(f"{marginalizacionElemP=}")
        print()
        

        tensor_result = tensor_product(
            marginalizacionElem.values, marginalizacionElemP.values
        )
        
        print(f"{tensor_result=}")
        print()
        
        tensor_result_2 = np.array(tensor_result).astype(np.float64)
        print(f"{initial_state_values=}")
        print(f"{tensor_result_2=}")
        emd = EMD(tensor_result_2, initial_state_values)
        
        print(f"{emd=}")
        print()
        emd_results[tuple(elem)] = emd
    
    print(f"{emd_results=}")
    print()
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
    ] = np.loadtxt("system_values_4.csv", delimiter=",", skiprows=1, dtype=str)
    initial_state = initial_state_str.strip()
    candidate_system = candidate_system_str.strip()
    present_subsystem = present_subsystem_str.strip()
    future_subsystem = future_subsystem_str.strip()
    matrix_1, states = load_tpm_2("./red2/state_node_a.csv", len(candidate_system))
    matrix_2, states = load_tpm_2("./red2/state_node_b.csv", len(candidate_system))
    matrix_3, states = load_tpm_2("./red2/state_node_c.csv", len(candidate_system))
    matrix_4, states = load_tpm_2("./red2/state_node_d.csv", len(candidate_system))
    matrix_5, states = load_tpm_2("./red2/state_node_e.csv", len(candidate_system))
    matrix_6, states = load_tpm_2("./red2/state_node_f.csv", len(candidate_system))
    matrix_7, states = load_tpm_2("./red2/state_node_g.csv", len(candidate_system))
    matrix_8, states = load_tpm_2("./red2/state_node_h.csv", len(candidate_system))
    matrix_9, states = load_tpm_2("./red2/state_node_i.csv", len(candidate_system))
    matrix_10, states = load_tpm_2("./red2/state_node_j.csv", len(candidate_system))
    
    tensor_flow = tensor_product_of_matrix(matrix_1, matrix_2)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_3)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_4)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_5)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_6)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_7)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_8)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_9)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_10)


    # print(tensor_flow)
    print(f"{initial_state=}, {candidate_system=}, {present_subsystem=}, {future_subsystem=}")
    df_tpm = apply_background(tensor_flow, initial_state, candidate_system)
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
    
    print("----------------------- MIN EMD ---------------------------")
    [min_emd_key, min_emd_result] = min_EMD(
        result_df.copy(), v.copy(), candidate_bipartitions, label
    )
    print(f"{min_emd_key=}, {min_emd_result=}")
    fin = time.perf_counter()
    print("Tiempo=")
    print(fin-inicio)

# casos de prueba primer excel 
def main_2():
    inicio = time.perf_counter()
    [
        initial_state_str,
        candidate_system_str,
        present_subsystem_str,
        future_subsystem_str,
    ] = np.loadtxt("system_values_2.csv", delimiter=",", skiprows=1, dtype=str)
    initial_state = initial_state_str.strip()
    candidate_system = candidate_system_str.strip()
    present_subsystem = present_subsystem_str.strip()
    future_subsystem = future_subsystem_str.strip()
    matrix_1, states = load_tpm_2("matrix_guia_2.csv", len(candidate_system))
    matrix_2, states = load_tpm_2("matrix_guia_3.csv", len(candidate_system))
    matrix_3, states = load_tpm_2("matrix_guia_4.csv", len(candidate_system))
    matrix_4, states = load_tpm_2("matrix_guia_5.csv", len(candidate_system))
    matrix_5, states = load_tpm_2("matrix_guia_6.csv", len(candidate_system))
    
    tensor_flow = tensor_product_of_matrix(matrix_1, matrix_2)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_3)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_4)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_5)

    # print(tensor_flow)
    print(f"{initial_state=}, {candidate_system=}, {present_subsystem=}, {future_subsystem=}")
    df_tpm = apply_background(tensor_flow, initial_state, candidate_system)
    print(df_tpm)

    v = build_v(present_subsystem, future_subsystem)
    print(f"{v=}")

    global global_v 

    global_v = v.copy()

    present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))
    result_df = marginalize_cols(df_tpm, future)
    result_df = marginalize_rows(result_df, present)

    node_states = get_matrices_node_state(result_df)
    for i in node_states:
        print("node_states=")
        print(node_states[i])

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
        result_df.copy(), v.copy(), candidate_bipartitions, label, node_states, initial_state
    )
    print(f"{min_emd_key=}, {min_emd_result=}")
    for i in node_states:
        print("node_states=")
        print(node_states[i])
    fin = time.perf_counter()
    print("Tiempo=")
    print(fin-inicio)
    

##############################################
# Caso de prueba red 15
##############################################
# inicio = time.perf_counter()
# [
#     initial_state_str,
#     candidate_system_str,
#     present_subsystem_str,
#     future_subsystem_str,
# ] = np.loadtxt("system_values_3.csv", delimiter=",", skiprows=1, dtype=str)
# initial_state = initial_state_str.strip()
# candidate_system = candidate_system_str.strip()
# present_subsystem = present_subsystem_str.strip()
# future_subsystem = future_subsystem_str.strip()
# matrix, states = load_tpm("resultado_15.csv", len(candidate_system))


# # print(tensor_flow)
# print(f"{initial_state=}, {candidate_system=}, {present_subsystem=}, {future_subsystem=}")
# df_tpm = apply_background(matrix, initial_state, candidate_system)
# # print(df_tpm)

# v = build_v(present_subsystem, future_subsystem)

# global global_v 

# global_v = v.copy()

# present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))
# result_df = marginalize_cols(df_tpm, future)
# result_df = marginalize_rows(result_df, present)

# node_states = get_matrices_node_state(result_df)


# candidates_bipartition = []
# candidate_bipartitions = bipartition_system(
#     result_df.copy(), v.copy(), initial_state, candidates_bipartition, node_states
# )
# print(f"{candidate_bipartitions=}")
# initial_state_v, _ = set_to_binary(global_v, v)
# present_idx = {idx: bit for idx, bit in enumerate(initial_state_v) if bit == "1"}
# sorted_idx = sorted(present_idx.keys())
# label = ""
# for idx in sorted_idx:
#     label += initial_state[idx]
    
# [min_emd_key, min_emd_result] = min_EMD(
#     result_df.copy(), v.copy(), candidate_bipartitions, label
# )
# print(f"{min_emd_key=}, {min_emd_result=}")
# fin = time.perf_counter()
# print("Tiempo=")
# print(fin-inicio)
main_2()