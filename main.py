import numpy as np
import pandas as pd
import string


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

    positions_to_keep = [i for i, bit in enumerate(present_subsystem) if bit == "1"]

    def extract_bits(binary_str, positions):
        return "".join([binary_str[i] for i in positions])

    new_index = df_tpm.index.map(lambda x: extract_bits(x, positions_to_keep))

    result_df = df_tpm.groupby(new_index).mean()
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


def tensor_product(df1: pd.DataFrame, df2: pd.DataFrame):
    result = pd.DataFrame()
    for df2col in df2.columns:
        for df1col in df1.columns:
            name = f"{df1col}{df2col}"
            result[name] = df1[df1col] * df2[df2col]

    return result


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
        -> marginarlizar por columnas A_t+1, B_t+1
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
def bipartition_system(df_tpm: pd.DataFrame, v: dict[str, str]):
    w_1 = {list(v.keys())[0]}
    keys = set(list(v.keys()))
    wp = keys - w_1

    results_u = {}
    while len(wp) > 0:
        for u in wp:
            w_1.add(u)
            w_1u = w_1.copy()
            w_1up = keys - w_1u

            print(w_1)
            print(w_1up)
            present, future = set_to_binary(v, w_1up, len(df_tpm.index[0]))
            marginalizacionW_1u = marginalize_cols(df_tpm, future)

            break
        break

def set_to_binary(v: dict[str, str], set: set[str], label_len:int):
    abc = string.ascii_lowercase
    binary_present = list(np.binary_repr(0, label_len))
    binary_future = list(np.binary_repr(0, label_len))
    for elem in set:
        idx = abc.index(elem[0])
        if 't+1' in elem:
            binary_future[idx] = '1'        
        else:
            binary_present[idx] = '1'
    
    return ["".join(binary_present), "".join(binary_future)]

def build_v(present_subsystem: str, future_subsystem: str):
    v = {}
    abc = string.ascii_lowercase
    for idx, bit in enumerate(present_subsystem):
        if bit == '1':
            v[f"{abc[idx]}_t"] = '1'
    
    for idx, bit in enumerate(future_subsystem):
        if bit == '1':
            v[f"{abc[idx]}_t+1"] = '1'
    
    return v

v = build_v(present_subsystem, future_subsystem)
bipartition_system(result_df, v)
# result_df = marginalize_rows(result_df, present_subsystem)
# print(result_df)

# df_a = marginalize_cols(result_df, "1000")
# df_b = marginalize_cols(result_df, "0100")
# df_c = marginalize_cols(result_df, "0010")

# result_ab = tensor_product(df_a, df_b)
# result_abc = tensor_product(result_ab, df_c)
# print(result_abc)
# result_abc = tensor_product(result_ab, df_c)
# print(result_abc)
