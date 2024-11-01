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
    sorted_index = sorted(result_df.index, key=lambda x: int(x[::-1], 2))
    result_df = result_df.reindex(sorted_index)
    # print(df_tpm.groupby(new_index).groups)
    return result_df


def marginalize_cols(df_tpm, future_subsystem: str):
    n_bits = len(df_tpm.columns[0])
    if len(future_subsystem) != n_bits:
        raise ValueError("invalid future subsystem")

    positions_to_keep = [i for i, bit in enumerate(future_subsystem) if bit == "1"]

    def extract_bits(binary_str, positions):
        return "".join([binary_str[i] for i in positions])

    new_index = df_tpm.columns.map(lambda x: extract_bits(x, positions_to_keep))
    result_df = df_tpm.T.groupby(new_index).sum()
    return result_df.T


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
    # result = pd.DataFrame()
    # for df2col in df2.columns:
    #     for df1col in df1.columns:
    #         name = f"{df1col}{df2col}"
    #         result[name] = df1[df1col] * df2[df2col]
    #         # print(df1[df1col] * df2[df2col])

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


def bipartition_system(df_tpm: pd.DataFrame, v: dict[str, str], initial_state: str):
    w_1 = {list(v.keys())[0]}
    keys = set(list(v.keys()))
    wp = keys - w_1
    results_u = {}

    initial_state_values = df_tpm.loc[initial_state, :].values
    while len(wp) > 0:
        for u in wp:
            w_1.add(u)
            w_1u = w_1.copy()
            w_1up = keys - w_1u

            print(f"w_1={w_1}")
            print(f"w_1u={w_1up}")
            """
                Necesito verificar el valor que tiene el estado inicial
                con el presente

                initial_state = 1000
                present       = 0100

            """

            # marginalization of w_1
            present, future = set_to_binary(v, w_1up, len(df_tpm.index[0]))
            print(f"present_w1u={present}")
            print(f"future_w1u={future}")
            present_idx = {idx: bit for idx, bit in enumerate(present) if bit == "1"}
            sorted_idx = sorted(present_idx.keys())
            label = ""
            for idx in sorted_idx:
                label += initial_state[idx]
            # print(
            #     f"present={present}, future={future}, w_1up={w_1up}, label={label}, present_idx={present_idx}"
            # )
            print()
            print(df_tpm)
            print()
            marginalizacionW_1u = marginalize_rows(df_tpm, present)
            print(marginalizacionW_1u)
            marginalizacionW_1u = marginalize_cols(marginalizacionW_1u, future)
            print(marginalizacionW_1u)
            # print()
            if len(label) > 0:
                marginalizacionW_1u = marginalizacionW_1u.loc[label, :]
            # print(marginalizacionW_1u.values)

            # marginalization of w_1up
            present, future = set_to_binary(v, w_1, len(df_tpm.index[0]))
            present_idx = {idx: bit for idx, bit in enumerate(present) if bit == "1"}
            sorted_idx = sorted(present_idx.keys())
            label = ""
            for idx in sorted_idx:
                label += initial_state[idx]
                
            print()
            print()
            
            print(f"present_w1={present}")
            print(f"future_w1={future}")
            # print(
            #     f"present={present}, future={future}, w_1={w_1}, label={label}, present_idx={present_idx}"
            # )

            marginalizacionW_1up = marginalize_rows(df_tpm, present)
            marginalizacionW_1up = marginalize_cols(marginalizacionW_1up, future)
            print(marginalizacionW_1up)
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
            up = keys - {u}
            present, future = set_to_binary(v, {u}, len(df_tpm.index[0]))
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
            present, future = set_to_binary(v, up, len(df_tpm.index[0]))
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
            
            results_u[u] = result_emd
            wp = wp - {u}
            w_1.add(u)
            break
        break
    
    # print(f"{results_u=}")
    # min_result = min(results_u.values())
    # print(f"{min_result}")


def set_to_binary(v: dict[str, str], set: set[str], label_len: int):
    abc = string.ascii_lowercase
    binary_present = list(np.binary_repr(0, label_len))
    binary_future = list(np.binary_repr(0, label_len))
    for elem in set:
        idx = abc.index(elem[0])
        if "t+1" in elem:
            binary_future[idx] = "1"
        else:
            binary_present[idx] = "1"

    return ["".join(binary_present), "".join(binary_future)]


def build_v(candidate_system: str):
    v = {}
    abc = string.ascii_lowercase
    for idx, bit in enumerate(candidate_system):
        if bit == "1":
            v[f"{abc[idx]}_t"] = "1"

    for idx, bit in enumerate(candidate_system):
        if bit == "1":
            v[f"{abc[idx]}_t+1"] = "1"

    return v


v = build_v(candidate_system)
bipartition_system(result_df, v, initial_state)
