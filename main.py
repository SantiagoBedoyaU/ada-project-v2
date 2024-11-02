import numpy as np
import pandas as pd
from pyemd import emd
from numpy.typing import NDArray
import string


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
    columns = pd.Index(
        [np.binary_repr(i, width=1)[::-1] for i in range(2)]
    )
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
    result_df = df_tpm.loc[filtered_states, filtered_states]
    return result_df


def marginalize_rows(df_tpm, present_subsystem: str):
    n_bits = len(df_tpm.index[0])
    if len(present_subsystem) != n_bits:
        raise ValueError("invalid present subsystem")

    positions_to_keep = [i for i, bit in enumerate(present_subsystem) if bit == "1"]

    def extract_bits(binary_str, positions):
        return "".join([binary_str[i] for i in positions])

    new_index = df_tpm.index.map(lambda x: extract_bits(x, positions_to_keep))
    result_df = df_tpm.groupby(new_index).mean()
    return reorder_little_endian(result_df)


def reorder_little_endian(df):
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


def marginalize_cols(df_tpm, future_subsystem: str):
    n_bits = len(df_tpm.columns[0])
    if len(future_subsystem) != n_bits:
        raise ValueError("invalid future subsystem")

    positions_to_keep = [i for i, bit in enumerate(future_subsystem) if bit == "1"]

    def extract_bits(binary_str, positions):
        return "".join([binary_str[i] for i in positions])

    new_index = df_tpm.columns.map(lambda x: extract_bits(x, positions_to_keep))
    result_df = df_tpm.T.groupby(new_index).sum()
    return reorder_little_endian(result_df.T)


def tensor_product(df1: list[float], df2: list[float]):
    if type(df1[0]) is np.ndarray:
        df1 = df1[0]
    if type(df2[0]) is np.ndarray:
        df2 = df2[0]

    # print(f"df1={df1}, df2={df2}")
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
            # print(df1[df1col] * df2[df2col])
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


[
    initial_state_str,
    candidate_system_str,
    present_subsystem_str,
    future_subsystem_str,
] = np.loadtxt("system_values.csv", delimiter=",", skiprows=1, dtype=str)
initial_state = initial_state_str.strip()
candidate_system = candidate_system_str.strip()
present_subsystem = present_subsystem_str.strip()
future_subsystem = future_subsystem_str.strip()

df_tpm, states = load_tpm("matrix_guia.csv", len(candidate_system))
# print(df_tpm)

result_df = apply_background(df_tpm, initial_state, candidate_system)
# print(result_df)

"""
    Particiones -> Esto es una funcion que recibe matriz despues de backgroud (result_df(#87)), recibe v
    present_subsystem = 1100 = A_t, B_t
    future_subsystem = 1100 =  A_t+1, B_t+1 

    v = [A_t, B_t, A_t+1, B_t+1] -> Sale del sistema candidato -> A_t+1 y B_t+1, es futuro del presete
    W_1 = [A_t] -> Es un elemento de B
    w' = v - W-1 = [B_t, A_t+1, B_t+1]

    resultados_por_u = {}
    -> Nota: la llave de resultados_por_u va a ser el elemento u que se itera y su valor
    va a ser el resultado de la resta de los EMD

    for u in w':
        iteracion 1 -> u = B_t
        Paso 1
        W_1u = W_1 + u = [A_t, B_t] -> Con estas son las que voy a trabajar, las de abajo se marginalizan
        W_1u' = [A_t+1, B_t+1]

        marginalizacionW_1u = [[]]
        -> marginarlizar por columnas A_t+1, B_t+1-
        -> Nota: Todo lo que no este en W_1u, se debe marginalizar, este decir
        se marginaliza las variables de W_1u'
        
        Paso 2
        -> Marginalizo las varibles de W_1u, es decir, trabajo con las variables
        de W_1u'
        marginalizacionW_1u' = [[]]

        Paso 3
        -> Producto tensorial entre marginalizacionW_1u y marginalizacionW_1u
        resultado_producto = [[]]

        Paso 4
        -> 1 EMD: Aplicar EMD (lo da la cucha) entre resultado_producto y matriz original (result_df)
        -> Nota: aplicar EMD en la fila donde este el estado incial, es decir
        si el estado inicial es 1000, entonces solo debo trabajar con la final que tenga este
        label tanto en resultado_producto como en la matriz original

        Paso 5
        u = B_t
        u' = [A_t, A_t+1, B_t+1]
        -> 1 Marginalizacion: Voy trabajar con u, es decir, voy a marginalizar las variables de u'
        -> 2 Margnalizacion: voy a trabajar con u', es decir, voy a marginalizar las variables de u

        -> Producto tensorial entre 1 marginalizacion y 2 marginalizacion
        -> 2 EMD: Aplicar EMD entre producto_tensorial (paso anterior) y matriz original
        -> Nota: aplicar EMD en la fila donde este el estado incial, es decir
        si el estado inicial es 1000, entonces solo debo trabajar con la final que tenga este
        label tanto en resultado_producto como en la matriz original
    
        Paso 6
        -> Restar al 1 EMD el 2 EMD -> 1EMD - 2EMD
        -> agregar el resultado de esta resta  a resultados_por_u

        -> Nota: Se hace lo mismo, para el seguiente u, es decir, A_t+1, y asi consecutivamente
    

    -> Despues de terminar de iterar, obtengo la llave que tenga el minimo valor de las resta
    de EMD; es decir, obtengo la llave en el que su valor sea el minimo entre todos los valores

    -> Esta llave que obtengo, lo debo agregar a W_1
    -> Esta llave, debe eliminarse de w'

    -> Importante: Desde la linea 102 (inicio del for) hasta la linea 151, este proceso
    se repite mientras w' tenga elementos
    
"""


def bipartition_system(
    df_tpm: pd.DataFrame, v: list, initial_state: str, candidates_bipartition: list
):
    if len(v) <= 2:
        candidates_bipartition.append(v[-1])
        return candidates_bipartition
    w_1 = [v[0]]
    w_1l = []
    wp = [item for item in v if item not in w_1]

    results_u = {}

    initial_state_values = df_tpm.loc[initial_state, :].values
    while len(wp) > 0:
        for u in wp:
            # print(w_1)
            w_1u = w_1.copy()
            w_1u.append(u)
            w_1up = [item for item in v if item not in w_1u]

            # print(f"w_1u={w_1u}")
            # print(f"w_1up={w_1up}")
            """
                Necesito verificar el valor que tiene el estado inicial
                con el presente

                initial_state = 1000
                present       = 0100

            """

            # marginalization of w_1
            present, future = set_to_binary(w_1up, len(df_tpm.index[0]))
            # print(f"present_w1up={present}")
            # print(f"future_w1up={future}")
            present_idx = {idx: bit for idx, bit in enumerate(present) if bit == "1"}
            sorted_idx = sorted(present_idx.keys())
            label = ""
            for idx in sorted_idx:
                label += initial_state[idx]
            # print(
            #     f"present={present}, future={future}, w_1up={w_1up}, label={label}, present_idx={present_idx}"
            # )
            # print()
            # print(df_tpm)
            # print()
            marginalizacionW_1u = marginalize_rows(df_tpm, present)
            # print(marginalizacionW_1u)
            marginalizacionW_1u = marginalize_cols(marginalizacionW_1u, future)
            # print(marginalizacionW_1u)
            # print()
            if len(label) > 0:
                marginalizacionW_1u = marginalizacionW_1u.loc[label, :]
            # print(marginalizacionW_1u.values)

            # marginalization of w_1up
            present, future = set_to_binary(w_1u, len(df_tpm.index[0]))
            present_idx = {idx: bit for idx, bit in enumerate(present) if bit == "1"}
            sorted_idx = sorted(present_idx.keys())
            label = ""
            for idx in sorted_idx:
                label += initial_state[idx]

            # print()
            # print()

            # print(f"present_w1u={present}")
            # print(f"future_w1u={future}")
            # print(
            #     f"present={present}, future={future}, w_1={w_1}, label={label}, present_idx={present_idx}"
            # )

            marginalizacionW_1up = marginalize_rows(df_tpm, present)
            marginalizacionW_1up = marginalize_cols(marginalizacionW_1up, future)
            # print(marginalizacionW_1up)
            # print()
            if len(label) > 0:
                marginalizacionW_1up = marginalizacionW_1up.loc[label, :]
            # print(marginalizacionW_1up.values)

            # tensor_product
            first_product_result = tensor_product(
                marginalizacionW_1u.values, marginalizacionW_1up.values
            )
            # print(f"tensor_product={first_product_result}")

            # 1 EMD
            first_product_result = np.array(first_product_result).astype(np.float64)
            initial_state_values = np.array(initial_state_values).astype(np.float64)
            # print(first_product_result)
            # print(initial_state_values)

            emd1 = EMD(first_product_result, initial_state_values)
            # print(emd1)

            # print(f"keys={keys}, u={u}")
            up = [item for item in v if item not in [u]]
            # print([u])
            # print(up)
            present, future = set_to_binary([u], len(df_tpm.index[0]))
            present_idx = {idx: bit for idx, bit in enumerate(present) if bit == "1"}
            sorted_idx = sorted(present_idx.keys())
            label = ""
            for idx in sorted_idx:
                label += initial_state[idx]
            # print(f"u={u}, present={present}, future={future}, label={label}")
            marginalizacionU = marginalize_rows(df_tpm, present)
            marginalizacionU = marginalize_cols(marginalizacionU, future)
            # print(marginalizacionU)
            if len(label) > 0:
                marginalizacionU = marginalizacionU.loc[label, :]

            # print()
            present, future = set_to_binary(up, len(df_tpm.index[0]))
            present_idx = {idx: bit for idx, bit in enumerate(present) if bit == "1"}
            sorted_idx = sorted(present_idx.keys())
            label = ""
            for idx in sorted_idx:
                label += initial_state[idx]
            # print(f"u={up}, present={present}, future={future}, label={label}")
            marginalizacionUp = marginalize_rows(df_tpm, present)
            marginalizacionUp = marginalize_cols(marginalizacionUp, future)
            # print(marginalizacionUp)
            if len(label) > 0:
                marginalizacionUp = marginalizacionUp.loc[label, :]

            second_product_result = tensor_product(
                marginalizacionU.values, marginalizacionUp.values
            )
            # print(f"{second_product_result=}")

            second_product_result = np.array(second_product_result).astype(np.float64)
            emd2 = EMD(second_product_result, initial_state_values)

            # print(f"{emd1=}, {emd2=}")
            result_emd = emd1 - emd2
            # diferencia_str = format(result_emd, '.20f')
            # print("Diferencia:", diferencia_str)

            if isinstance(u, list):
                results_u[tuple(u)] = result_emd
            else:
                results_u[u] = result_emd

        # print(f"{results_u=}")
        min_result = min(results_u.values())
        # print(f"{min_result=}")
        key = [key for key, value in results_u.items() if value == min_result][0]
        # print(f"{key=}")
        if isinstance(key, tuple):
            key = list(key)
        wp.remove(key)
        w_1.append(key)
        w_1l.append(key)
        # print(results_u)
        # print(w_1)
        # print(w_1l)
        # print()

        # print(w_1l)
        results_u.clear()
        # print(f"{w_1=}")
        # print(f"{w_1l=}")
        # print()
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
    # print(v)
    # print(w_1l)
    candidates_bipartition = bipartition_system(
        df_tpm, v, initial_state, candidates_bipartition
    )
    return candidates_bipartition


def set_to_binary(set: list, label_len: int):
    abc = string.ascii_lowercase
    binary_present = list(np.binary_repr(0, label_len))
    binary_future = list(np.binary_repr(0, label_len))

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

def get_matrices_node_state(df_tpm: pd.DataFrame):
    matrices_node_state = {}
    string = '0' * len(df_tpm.index[0])
    for i in range(len(df_tpm.index[0])):
        future = string[:i] + '1' + string[i+1:]
        matrix = marginalize_cols(df_tpm.copy(), future)
        matrices_node_state[future] = matrix
    return matrices_node_state

def build_v(candidate_system: str):
    v = []
    abc = string.ascii_lowercase
    for idx, bit in enumerate(candidate_system):
        if bit == "1":
            v.append(f"{abc[idx]}_t")

    for idx, bit in enumerate(candidate_system):
        if bit == "1":
            v.append(f"{abc[idx]}_t+1")

    return v

def build_v_2(present_subsystem: str, future_subsystem: str):
    v = []
    abc = string.ascii_lowercase
    for idx, bit in enumerate(present_subsystem):
        if bit == "1":
            v.append(f"{abc[idx]}_t")

    for idx, bit in enumerate(future_subsystem):
        if bit == "1":
            v.append(f"{abc[idx]}_t+1")

    return v

def min_EMD(df_tpm: pd.DataFrame, v: list[str], bipartion_list: list[str], initial_state: str):
    initial_state_values = df_tpm.loc[initial_state, :].values
    initial_state_values = np.array(initial_state_values).astype(np.float64)

    emd_results = {}
    for elem in bipartion_list:
        # print(f"{elem=}")
        elemP = []
        if isinstance(elem, list):
            for e in v:
                if e not in elem:
                    elemP.append(e)
        else:
            elemP = [e for e in v if e != elem]
        # print(f"{elemP=}")
        
        presentElem, futureElem = set_to_binary([elem], len(df_tpm.index[0]))
        present_idx = {idx: bit for idx, bit in enumerate(presentElem) if bit == "1"}
        sorted_idx = sorted(present_idx.keys())
        label = ""
        for idx in sorted_idx:
            label += initial_state[idx]
        # print(f"{presentElem=}, {futureElem=}")
        marginalizacionElem = marginalize_rows(df_tpm, presentElem)
        marginalizacionElem = marginalize_cols(marginalizacionElem, futureElem)
        if len(label) > 0:
            marginalizacionElem = marginalizacionElem.loc[label, :]

        # print(f"{marginalizacionElem}")

        presentElemP, futureElemP = set_to_binary([elemP], len(df_tpm.index[0]))
        present_idx = {idx: bit for idx, bit in enumerate(presentElemP) if bit == "1"}
        sorted_idx = sorted(present_idx.keys())
        label = ""
        for idx in sorted_idx:
            label += initial_state[idx]
        # print(f"{presentElemP=}, {futureElemP=}")
        marginalizacionElemP = marginalize_rows(df_tpm, presentElemP)
        marginalizacionElemP = marginalize_cols(marginalizacionElemP, futureElemP)
        if len(label) > 0:
            marginalizacionElemP = marginalizacionElemP.loc[label, :]
        # print(f"{marginalizacionElemP=}")
        
        tensor_result = tensor_product(marginalizacionElem.values, marginalizacionElemP.values)
        tensor_result = np.array(tensor_result).astype(np.float64)
        emd = EMD(tensor_result, initial_state_values)
        emd_results[tuple(elem)] = emd
    # print(emd_results)
    min_emd_result = min(emd_results.values())
    min_emd_key = [key for key, value in emd_results.items() if value == min_emd_result][0]
    return [min_emd_key, min_emd_result]


v = build_v(candidate_system)
print(f"{v=}")
candidates_bipartition = []
# print(result_df)
candidate_bipartitions = bipartition_system(result_df.copy(), v.copy(), initial_state, candidates_bipartition)
print(f"{candidate_bipartitions=}")
[min_emd_key, min_emd_result] = min_EMD(result_df, v, candidate_bipartitions, initial_state)
print(f"{min_emd_key=}, {min_emd_result=}")

# f = []
# example = ['1','2', ['1','2'], ['1', '2']]
# f.append(example[-2:])

# print(f)


##############################################
# Casos de prueba
##############################################

# [
#     initial_state_str,
#     candidate_system_str,
#     present_subsystem_str,
#     future_subsystem_str,
# ] = np.loadtxt("system_values_2.csv", delimiter=",", skiprows=1, dtype=str)
# initial_state = initial_state_str.strip()
# candidate_system = candidate_system_str.strip()
# # print(candidate_system)
# present_subsystem = present_subsystem_str.strip()
# future_subsystem = future_subsystem_str.strip()
# matrix_1, states = load_tpm_2('matrix_guia_2.csv', len(candidate_system))
# matrix_2, states  = load_tpm_2('matrix_guia_3.csv', len(candidate_system))
# matrix_3, states  = load_tpm_2('matrix_guia_4.csv', len(candidate_system))
# matrix_4, states  = load_tpm_2('matrix_guia_5.csv', len(candidate_system))
# matrix_5, states  = load_tpm_2('matrix_guia_6.csv', len(candidate_system))
# tensor_flow = tensor_product_of_matrix(matrix_1, matrix_2)
# tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_3)
# tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_4)
# tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_5)


# df_tpm= apply_background(tensor_flow, initial_state, candidate_system)
# print(df_tpm)

# node_states = get_matrices_node_state(df_tpm)
# print(node_states['00001'])

# v = build_v_2(present_subsystem, future_subsystem)
# print(v)

# Marginalizacion a los subsistemas:
# df_tpm= marginalize_cols(df_tpm, '11011')
# print(df_tpm)

# candidates_bipartition = []
# candidate_bipartitions = bipartition_system(df_tpm.copy(), v.copy(), initial_state, candidates_bipartition)
# print(f"{candidate_bipartitions=}")
# [min_emd_key, min_emd_result] = min_EMD(df_tpm.copy(), v, candidate_bipartitions, initial_state)
# print(f"{min_emd_key=}, {min_emd_result=}")
