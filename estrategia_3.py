from fastapi import UploadFile
from pyemd import emd
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import string
import time
from threading import Thread
from icecream import ic


def load_tpm(filename_tpm: str, num_elements: int):
    """
    Carga una matriz de transición de probabilidad (TPM) desde un archivo y genera un índice de estados.

    Args:
        filename_tpm (str): La ruta al archivo que contiene la matriz de transición de probabilidad.
        num_elements (int): El número de elementos (o estados) que se utilizarán para generar el índice.

    Returns:
        tuple: Una tupla que contiene:
            - pd.DataFrame: Un DataFrame que representa la matriz de transición de probabilidad, 
              con índices y columnas correspondientes a los estados.
            - pd.Index: Un índice de estados generado a partir del número de elementos.

    Raises:
        ValueError: Si el archivo no se puede cargar o si el número de elementos es inválido.
    
    Example:
        >>> df, states = load_tpm("tpm.csv", 3)
        >>> print(df)
    """
    states = pd.Index(
        [np.binary_repr(i, width=num_elements)[::-1] for i in range(2**num_elements)]
    )
    tpm = np.loadtxt(filename_tpm, delimiter=",")
    df_tpm = pd.DataFrame(tpm, index=states, columns=states)
    return df_tpm, states

def load_tpm_2(filename_tpm: str, num_elements: int):
    """
    Carga una matriz de transición de probabilidad (TPM) desde un archivo y genera un índice de estados.
    Sirve para cargar matrices nodo-estado.

    Args:
        filename_tpm (str): La ruta al archivo que contiene la matriz de transición de probabilidad.
        num_elements (int): El número de elementos (o estados) que se utilizarán para generar el índice.

    Returns:
        tuple: Una tupla que contiene:
            - pd.DataFrame: Un DataFrame que representa la matriz de transición de probabilidad, 
              con índices y columnas correspondientes a los estados.
            - pd.Index: Un índice de estados generado a partir del número de elementos.

    Raises:
        ValueError: Si el archivo no se puede cargar o si el número de elementos es inválido.
    
    Example:
        >>> df, states = load_tpm("tpm.csv", 3)
        >>> print(df)
    """
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
    """
    Aplica condiciones de background a una matriz de transición de probabilidad (TPM) 
    y genera un nuevo DataFrame marginalizado.

    Args:
        df_tpm (pd.DataFrame): DataFrame que representa la matriz de transición de probabilidad.
        initial_state: Estado inicial que se utilizará para filtrar los estados.
        candidate_system: Sistema candidato que determina qué bits se consideran.

    Returns:
        pd.DataFrame: Un nuevo DataFrame que contiene los estados filtrados y marginalizados.

    Example:
        >>> result_df = apply_background(df_tpm, '1010', '1110')
    """
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
    """
    Marginaliza las filas de una matriz de transición de probabilidad (TPM) 
    según un subsistema presente, calculando el promedio de la suma de las filas.

    Args:
        df_tpm (pd.DataFrame): DataFrame que representa la matriz de transición de probabilidad.
        present_subsystem (str): Cadena binaria que indica que estados se deben mantener.

    Returns:
        pd.DataFrame: Un nuevo DataFrame que contiene las filas marginalizadas.

    Raises:
        ValueError: Si la longitud del subsistema presente no coincide con el número de estados presente en la tpm.

    Example:
        >>> result_df = marginalize_rows(df_tpm, "110")
    """
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
    """
    Reordena las filas y columnas de un DataFrame según el orden little-endian 
    basado en sus índices y nombres de columnas, respectivamente.

    Args:
        df (pd.DataFrame): DataFrame que se desea reordenar.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con las filas y columnas reordenadas.

    Example:
        >>> reordered_df = reorder_little_endian(df)
    """
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
    """
    Marginaliza las columnas de una matriz de transición de probabilidad (TPM) 
    según un subsistema futuro, sumando los estados futuros.

    Args:
        df_tpm (pd.DataFrame): DataFrame que representa la matriz de transición de probabilidad.
        future_subsystem (str): Cadena binaria que indica qué estados futuro se deben mantener.

    Returns:
        pd.DataFrame: Un nuevo DataFrame que contiene las columnas marginalizadas.

    Raises:
        ValueError: Si la longitud del subsistema futuro no coincide con el número de estados futuro en la TPM.

    Example:
        >>> result_df = marginalize_cols(df_tpm, "110")
    """
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
    """
    Calcula el producto tensorial de dos DataFrames, combinando sus índices y valores 
    según las claves especificadas.

    Args:
        df1 (pd.DataFrame): Primer DataFrame para el producto tensorial.
        df2 (pd.DataFrame): Segundo DataFrame para el producto tensorial.
        keys_df1 (list): Lista de claves que representan los índices de df1.
        keys_df2 (list): Lista de claves que representan los índices de df2.

    Returns:
        pd.DataFrame: Un nuevo DataFrame que representa el producto tensorial de df1 y df2.

    Example:
        >>> result_df = tensor_product(df1, df2, ['a', 'b'], ['c', 'd'])
    """
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
    """
    Calcula el producto tensorial de dos DataFrames, combinando columnas de cada uno.
    Esta función se usa cuando el producto tensorial debe hacerse entre matrices con más 
    de una fila.

    Args:
        df1 (pd.DataFrame): Primer DataFrame para el producto tensorial.
        df2 (pd.DataFrame): Segundo DataFrame para el producto tensorial.

    Returns:
        pd.DataFrame: Un nuevo DataFrame que representa el producto tensorial de df1 y df2.

    Example:
        >>> result_df = tensor_product_of_matrix(df1, df2)
    """
    # Diccionario para almacenar columnas temporalmente
    result_dict = {}
    for df2col in df2.columns:
        for df1col in df1.columns:
            name = f"{df1col}{df2col}"
            result_dict[name] = df1[df1col] * df2[df2col]

    # Crear el dataframe de una vez usando pd.DataFrame
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
    """
    Marginaliza las nodo-estado de una matriz de transición de probabilidad (TPM) 
    y combina los resultados en un solo DataFrame.

    Args:
        df_tpm (pd.DataFrame): DataFrame que representa la matriz de transición de probabilidad.
        node_state (dict): Diccionario que contiene los estados de los nodos.
        set_m (list[str]): Lista de elementos que indican a quién marginalizar.
        label (str): Estado inicial.

    Returns:
        pd.DataFrame: Un DataFrame que representa el estado marginalizado del nodo especificado.

    Example:
        >>> result = marginalize_node_states(df_tpm, node_state, set_m, label)
    """
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
    """
    Realiza la marginalización de las matrices nodo-estado en una matriz de transición de probabilidad 
    (TPM) y combina los resultados en un solo DataFrame.

    Args:
        present (str): Estado presente que se utilizará para la marginalización.
        node_state (dict): Diccionario que contiene las matrices nodo-estado.
        set_m (list): Lista de elementos para marginalizar.

    Returns:
        pd.DataFrame: Un DataFrame que representa el estado marginalizado de las nodo-estado.

    Example:
        >>> result = first_marginalize_node_states(present, node_state, set_m)
    """
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
    """
    Realiza la búsqueda de la bipartición de un sistema basado en la matriz de transición de probabilidad 
    (TPM) y sus matrices nodo-estado, buscando encontrar una aproximación a la menor perdida de información.

    Args:
        df_tpm (pd.DataFrame): DataFrame que representa la matriz de transición de probabilidad.
        v (list): Lista de nodos del sistema completo.
        initial_state (str): Estado inicial de la TPM.
        candidates_bipartition (list): Lista que contendrá la mejor bipartición candidata.
        node_state (dict): Diccionario que contiene las matrices nodo-estado.

    Returns:
        list: Una lista actualizado de la mejor bipartición encontrada.

    Example:
        >>> candidates = bipartition_system(df_tpm, a, initial_state, candidates_bipartition, node_state)
    """
    # First config
    v_copy = v.copy()
    n = int(len(v_copy) / 2)
    w_1u = np.random.choice(v_copy, size=n, replace=False)
    n_fails = 50 * len(v_copy)
    ic(n_fails, len(v_copy))

    # --------------- INITIAL_STATE ------------------
    present_v, _, _ = set_to_binary(global_v, v)
    present_idx = {idx: bit for idx, bit in enumerate(present_v) if bit == "1"}
    sorted_idx = sorted(present_idx.keys())
    label = ""
    for idx in sorted_idx:
        label += initial_state[idx]
    
    initial_state_values = df_tpm.loc[label, :].values

    request_threads = {}
    
    emd1 = calculate_bipartition_emd(df_tpm, v, w_1u, node_state, initial_state, initial_state_values)
    candidates_bipartition[0] = [w_1u, emd1]
    while n_fails > 0:
        if n_fails % len(v_copy) == 0 and n > 1:
            n -=1
        w_1up = np.random.choice(v_copy, size=n, replace=False)
        key = tuple(w_1up)
        request_threads[key] = Thread(target=calculate_bipartition_emd, kwargs={
            'df_tpm': df_tpm,
            'v': v,
            'subset': w_1up,
            'node_state': node_state,
            'initial_state': initial_state,
            'initial_state_values': initial_state_values,
        })
        request_threads[key].start()
        n_fails-=1
        # emd1_p = calculate_bipartition_emd(df_tpm, v, w_1up, node_state, initial_state, initial_state_values)
        # result_emd = emd1_p - emd1
        # if result_emd < 0:
        #     candidates_bipartition[0] = [w_1up, emd1_p]
        #     emd1 = emd1_p
        # else:
        #     n_fails-=1

    while True:
        request_threads = clean_old_request_threads(request_threads)
        if len(request_threads) < 1:
            break
    # return candidates_bipartition

def clean_old_request_threads(request_threads):
    for key, value in request_threads.copy().items():
        if not value.is_alive():
            del request_threads[key]

    return request_threads

def calculate_bipartition_emd(df_tpm, v, subset, node_state, initial_state, initial_state_values):
    tuple_subset = tuple(subset)
    if tuple_subset in bipartition_tpm:
        return bipartition_tpm[tuple_subset]
    else:
        # --------------- RAMDOM_TWO_NODES ----------------
        w_1up = [item for item in v if item not in subset]
        #----------------MARGINALIZACION W_1U ----------------
        present, future, set_m_w1 = set_to_binary(global_v, subset)
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
        emd = EMD(first_product_result, initial_state_values)
        bipartition_tpm[tuple_subset] = emd

def set_to_binary_1(set: list, present_label_len: int, future_label_len: int):
    """
    Convierte un conjunto de elementos en representaciones binarias para los estados presentes 
    y futuros basados en la longitud de los nodos proporcionados.

    Args:
        set (list): Lista de elementos a convertir en binario. Los elementos pueden ser cadenas 
        o listas de cadenas que contengan nodos.
        present_label_len (int): Longitud de la representación binaria para el estado presente.
        future_label_len (int): Longitud de la representación binaria para el estado futuro.

    Returns:
        list: Una lista que contiene dos cadenas binarias: la primera para el estado presente 
              y la segunda para el estado futuro.

    Example:
        >>> binary_states = set_to_binary_1(['a', 'b', 'c_t+1'], 3, 3)
        >>> print(binary_states)  # Output: ['110', '001']
    """
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
    """
    Convierte un elemento y un conjunto global en representaciones binarias para los estados 
    presentes y futuros, excluyendo el elemento especificado. Esta función se usa para representar
    el estado binario de un nodo con respecto a todos los demás nodos

    Args:
        global_a (list): Lista de pares de elementos, donde el primer elemento representa 
        el estado presente y el segundo el estado futuro.
        element (str): Un elemento en forma de cadena que se debe excluir del conjunto global.

    Returns:
        tuple: Una tupla que contiene dos cadenas binarias: la primera para el estado presente 
               y la segunda para el estado futuro de la arista.

    Example:
        >>> binary_states = set_to_binary(['at', 'bt', 'ct', 'at+1', 'bt+1', 'ct+1'], 'at+1')
        >>> print(binary_states)  # Output: ('100', '000')
    """
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
    """
    Genera matrices nodos-estado a partir de un DataFrame de transición de probabilidad 
    y un subsistema futuro especificado.

    Args:
        df_tpm (pd.DataFrame): DataFrame que representa la matriz de transición de probabilidad.
        future_subsystem (list): Lista que indica el estado futuro de cada nodo, donde '1' 
                                 representa un estado activo.

    Returns:
        dict: Un diccionario donde las claves son letras (a, b, c, ...) y los valores son 
              matrices de estado correspondientes a cada nodo activo.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame([[0.1, 0.9], [0.4, 0.6]], columns=['A', 'B'])
        >>> states = get_matrices_node_state(df, ['1', '0'])
        >>> print(states)  # Output: {'a': matrix_for_a}
    """
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
    """
    Genera matrices nodo-estado inicial del sistema completo. Esta función solo se usa para el momento
    de marginalización de los subsistemas presente y futuro.

    Args:
        df_tpm (pd.DataFrame): DataFrame que representa la matriz de transición de probabilidad.

    Returns:
        dict: Un diccionario donde las claves son letras (a, b, c, ...) y los valores son 
              matrices de estado correspondientes a cada nodo.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame([[0.1, 0.9], [0.4, 0.6]], columns=['A', 'B'])
        >>> states = get_first_matrices_node_state(df)
        >>> print(states)  # Output: {'a': matrix_for_a, 'b': matrix_for_b, ...}
    """
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
    """
    Construye una lista de nodos representados en letras a partir de los subsistemas presentes y futuros
    representados en binarios.

    Args:
        present_subsystem (str): Cadena que representa el estado presente, donde '1' indica 
                                 un estado activo.
        future_subsystem (str): Cadena que representa el estado futuro, donde '1' indica 
                                un estado activo.

    Returns:
        list: Lista de variables de estado en el formato 'x_t' y 'x_t+1', donde 'x' es la 
              letra correspondiente al estado activo.

    Example:
        >>> v = build_v('110', '001')
        >>> print(v)  # Output: ['a_t', 'b_t', 'c_t+1']
    """
    v = []
    abc = string.ascii_lowercase
    for idx, bit in enumerate(present_subsystem):
        if bit == "1":
            v.append(f"{abc[idx]}_t")
    for idx, bit in enumerate(future_subsystem):
        if bit == "1":
            v.append(f"{abc[idx]}_t+1")
    return v

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
    print(f"{initial_state=}, {candidate_system=}, {present_subsystem=}, {future_subsystem=}")
    matrix_1, _ = load_tpm_2("./red2/state_node_a.csv", len(candidate_system))
    matrix_2, _ = load_tpm_2("./red2/state_node_b.csv", len(candidate_system))
    matrix_3, _ = load_tpm_2("./red2/state_node_c.csv", len(candidate_system))
    matrix_4, _ = load_tpm_2("./red2/state_node_d.csv", len(candidate_system))
    matrix_5, _ = load_tpm_2("./red2/state_node_e.csv", len(candidate_system))
    matrix_6, _ = load_tpm_2("./red2/state_node_f.csv", len(candidate_system))
    matrix_7, _ = load_tpm_2("./red2/state_node_g.csv", len(candidate_system))
    matrix_8, _ = load_tpm_2("./red2/state_node_h.csv", len(candidate_system))
    matrix_9, _ = load_tpm_2("./red2/state_node_i.csv", len(candidate_system))
    matrix_10, _ = load_tpm_2("./red2/state_node_j.csv", len(candidate_system))
    
    tensor_flow = tensor_product_of_matrix(matrix_1, matrix_2)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_3)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_4)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_5)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_6)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_7)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_8)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_9)
    tensor_flow = tensor_product_of_matrix(tensor_flow, matrix_10)
    
    inicio = time.perf_counter()

    df_tpm = apply_background(tensor_flow, initial_state, candidate_system)
    
    v = build_v(present_subsystem, future_subsystem)

    global global_v 
    global_v = v.copy()
    global marginalized_tpm
    marginalized_tpm = {}
    global bipartition_tpm
    bipartition_tpm = {}
    
    present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))

    node_states = get_first_matrices_node_state(df_tpm)

    result_df = marginalize_node_states_1(df_tpm, present, future, node_states, sorted(node_states.keys()))
    result_df = marginalize_cols(result_df, future)
    node_states = get_matrices_node_state(result_df, future)
    
    candidates_bipartition = [0]
    candidate_bipartitions = bipartition_system(
        result_df.copy(), v.copy(), initial_state, candidates_bipartition, node_states
    )
    print(candidate_bipartitions[0])
    fin = time.perf_counter()
    print("Tiempo=")
    print(fin-inicio)

# casos de prueba primer excel 
def main_2():
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
    matrix_1, _ = load_tpm_2("matrix_guia_2.csv", len(candidate_system))
    matrix_2, _ = load_tpm_2("matrix_guia_3.csv", len(candidate_system))
    matrix_3, _ = load_tpm_2("matrix_guia_4.csv", len(candidate_system))
    matrix_4, _ = load_tpm_2("matrix_guia_5.csv", len(candidate_system))
    matrix_5, _ = load_tpm_2("matrix_guia_6.csv", len(candidate_system))
    
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
    global bipartition_tpm
    bipartition_tpm = {}
    
    present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))

    node_states = get_first_matrices_node_state(df_tpm)

    result_df = marginalize_node_states_1(df_tpm, present, future, node_states, sorted(node_states.keys()))
    result_df = marginalize_cols(result_df, future)
    node_states = get_matrices_node_state(result_df, future)
    
    candidates_bipartition = [0]
    inicio = time.perf_counter()
    candidate_bipartitions = bipartition_system(
        result_df.copy(), v.copy(), initial_state, candidates_bipartition, node_states
    )
    ic(bipartition_tpm)
    # print(candidate_bipartitions[0])
    bestEMD = sorted(bipartition_tpm.values()).pop(0)
    value = {i for i in bipartition_tpm if bipartition_tpm[i] == bestEMD}
    ic(value, len(value))
    ic(bestEMD)
    
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
    global bipartition_tpm
    bipartition_tpm = {}

    present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))

    node_states = get_first_matrices_node_state(df_tpm)

    result_df = marginalize_node_states_1(df_tpm, present, future, node_states, sorted(node_states.keys()))
    result_df = marginalize_cols(result_df, future)
    node_states = get_matrices_node_state(result_df, future)

    inicio = time.perf_counter()
    candidates_bipartition = [0]
    candidate_bipartitions = bipartition_system(
        result_df.copy(), v.copy(), initial_state, candidates_bipartition, node_states
    )
    fin = time.perf_counter()
    bestEMD = sorted(bipartition_tpm.values()).pop(0)
    value = {i for i in bipartition_tpm if bipartition_tpm[i] == bestEMD}
    ic(value, len(value))
    ic(bestEMD)
    return [value, bestEMD, fin-inicio]
# main_2()