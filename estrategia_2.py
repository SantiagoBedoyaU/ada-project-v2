from fastapi import UploadFile
import numpy as np
import pandas as pd
from pyemd import emd
from numpy.typing import NDArray
import string
import time

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

def marginalize_rows(df_tpm: pd.DataFrame, present_subsystem: str):
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
        future_subsystem (str): Cadena binaria que indica qué estadoss futuro se deben mantener.

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

def expand_matrix(df_tpm: pd.DataFrame, present: str):
    """
    Expande una matriz de transición de probabilidad (TPM) agregando un estado siguiendo 
    el orden little endian, según un patrón binario que indica cómo deben ser modificados los índices.

    Args:
        df_tpm (pd.DataFrame): DataFrame que representa la matriz de transición de probabilidad.
        present (str): Cadena binaria que indica qué estados expandir(los puestos en '0').

    Returns:
        pd.DataFrame: Un nuevo DataFrame que representa la matriz TPM expandida.

    Example:
        >>> expanded_df = expand_matrix(df_tpm, "101")
    """
    empty_label = [0] * len(present)
    label_results = []
    new_labels = []
    expanded_df_tpm = df_tpm
    for idx, digit in enumerate(present):
        if digit == '0':
            empty_label[idx] = ' '
        else:
            empty_label[idx] = '-1'
    
    label = empty_label
    for idx, digit in enumerate(present):
        if digit == '0':
            step = 2 ** idx
            expanded_df_tpm = interleave(expanded_df_tpm, step)
            # expantion
            assignment = '1'
            cont = 0
            for lab in expanded_df_tpm.index:
                if cont % step == 0:
                    assignment = '0' if assignment == '1' else '1'                
                index_lab = 0
                for value in label:
                    if value == '-1':
                        label_results.append(lab[index_lab]) 
                        index_lab += 1
                    else:
                        label_results.append(value)
                label_results[label_results.index(' ')] = assignment
                label_results = [elem for elem in label_results if elem != ' ']
                
                new_labels.append("".join(label_results))
                label_results.clear()
                cont += 1
            expanded_df_tpm.index = new_labels
            new_labels.clear()
            label[label.index(' ')] = '-1'
    return expanded_df_tpm
            
def interleave(df, n):
    """
    Intercala las filas de un DataFrame en bloques de tamaño n.

    Args:
        df (pd.DataFrame): DataFrame cuyas filas se van a intercalar.
        n (int): Tamaño del bloque para la intercalación.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con las filas intercaladas.

    Example:
        >>> interleaved_df = interleave(df, 2)
    """
    interleaved_rows = []
    i1, i2 = 0, 0
    
    while i1 < len(df) or i2 < len(df):
        for _ in range(n):
            if i1 < len(df):
                interleaved_rows.append(df.iloc[i1])
                i1 += 1
        
        for _ in range(n):
            if i2 < len(df):
                interleaved_rows.append(df.iloc[i2])
                i2 += 1
    return pd.DataFrame(interleaved_rows, columns=df.columns)

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
    node_state: dict,
    set_m: list[str],
    label: str
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
    viewed_letters = []
    tuple_set_m = []
    results_node_states = node_state.copy()
    for sublist in set_m:
        if isinstance(sublist, list):
            tuple_set_m.append(tuple(sublist))
        else:
            tuple_set_m.append(sublist)
    tuple_set_m = tuple(tuple_set_m)
    if tuple_set_m in marginalized_tpm:
        return marginalized_tpm[tuple_set_m]
    else:
        for elem in set_m:
            if isinstance(elem, list):
                for el in elem:
                    present, _ = set_to_binary(global_a, el)
                    future_letter = el[1].lower()
                    if future_letter in viewed_letters:
                        a = results_node_states[future_letter]
                    else:
                        a = node_state[future_letter]
                        viewed_letters.append(future_letter)
                    result_a = marginalize_rows(a.copy(), present)
                    result_a = expand_matrix(result_a, present)
                    results_node_states[future_letter] = result_a
            else:
                present, _ = set_to_binary(global_a, elem)
                future_letter = elem[1].lower()
                if future_letter in viewed_letters:
                    a = results_node_states[future_letter]
                else:
                    a = node_state[future_letter]
                    viewed_letters.append(future_letter)
                result_a = marginalize_rows(a.copy(), present)
                result_a = expand_matrix(result_a, present)
                results_node_states[future_letter] = result_a
    
        keys = sorted(results_node_states.keys())
        for i in range(0, len(keys)):
            results_node_states[keys[i]] = results_node_states[keys[i]].loc[[label], :]
        if len(keys) > 0:
            first = keys[0]
        for i in range(1, len(keys)):
            results_node_states[keys[i]] = tensor_product(results_node_states[keys[i-1]], results_node_states[keys[i]], list(first), list(keys[i])) 
            first = "".join([first, keys[i]])
        marginalized_tpm[tuple_set_m] = results_node_states[keys[-1]] 
        return results_node_states[keys[-1]] 

def first_marginalize_node_states(
    present: str,
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
        result_a = marginalize_rows(a.copy(), present)
        results_node_states[elem] = result_a  
    
    marginalized_tpm.clear()
    keys = sorted(results_node_states.keys())
    if len(keys) > 0:
        first = keys[0]
    for i in range(1, len(keys)):
        results_node_states[keys[i]] = tensor_product_of_matrix(results_node_states[keys[i-1]], results_node_states[keys[i]]) 
        first = "".join([first, keys[i]])
    marginalizacion = results_node_states[keys[-1]]
    return marginalizacion

def bipartition_system(
    df_tpm: pd.DataFrame,
    a: list,
    initial_state: str,
    candidates_bipartition: dict,
    node_state: dict,
):
    """
    Realiza la búsqueda de la bipartición de un sistema basado en la matriz de transición de probabilidad 
    (TPM) y sus matrices nodo-estado, buscando minimizar la perdida de información.

    Args:
        df_tpm (pd.DataFrame): DataFrame que representa la matriz de transición de probabilidad.
        a (list): Lista de aristas del sistema completo.
        initial_state (str): Estado inicial de la TPM.
        candidates_bipartition (dict): Diccionario que contendrá la mejor bipartición candidata.
        node_state (dict): Diccionario que contiene las matrices nodo-estado.

    Returns:
        dict: Un diccionario actualizado de la mejor bipartición encontrada.

    Example:
        >>> candidates = bipartition_system(df_tpm, a, initial_state, candidates_bipartition, node_state)
    """
    if len(a) <= 2:
        return candidates_bipartition
    w_1 = [a[0]]
    w_1l = []
    wp = [item for item in a if item not in w_1]
    results_u = {}
    
    present_a, _, _ = set_to_binary_2(global_a, a)
    present_idx = {idx: bit for idx, bit in enumerate(present_a) if bit == "1"}
    sorted_idx = sorted(present_idx.keys())
    label = ""
    for idx in sorted_idx:
        label += initial_state[idx]
    initial_state_values = df_tpm.loc[label, :].values
    b_break = False
    while len(wp) > 0:
        if b_break:
            break
        for u in wp:
            w_1u = w_1.copy()
            w_1u.append(u)
            isParticioned = False
            adjacency_matrix = create_adjacency_matrix(w_1u)
            num_components, _ = count_bipartite_graph_components(adjacency_matrix)
            #----------------MARGINALIZACION W_1U ----------------
            marginalizacionW_1u = marginalize_node_states(
                node_state, w_1u, label
            )
            #----------------FIRST_EMD---------------------
            first_product_result = np.array(marginalizacionW_1u).flatten().astype(np.float64)
            initial_state_values = np.array(initial_state_values).astype(np.float64)
            emd1 = EMD(first_product_result, initial_state_values)
            #------------- MARGINALIZACION U ---------------
            marginalizacionU = marginalize_node_states(
                node_state, [u], label
            )
            #----------------SECOND_EMD---------------------
            second_product_result = np.array(marginalizacionU).flatten().astype(np.float64)
            emd2 = EMD(second_product_result, initial_state_values)
            result_emd = emd1 - emd2
            #----------------BIPARTITION-------------------
            if num_components == 2:
                isParticioned = True
                better_candidate = candidates_bipartition['0']
                if better_candidate[-1] == emd1 and len(better_candidate[0]) < len(w_1u):
                    candidates_bipartition['0'] = [w_1u, emd1] 
                elif better_candidate[-1] > emd1:
                    candidates_bipartition['0'] = [w_1u, emd1]

            if isinstance(u, list):
                results_u[tuple(u)] = [result_emd, emd1, isParticioned, num_components]
            else:
                results_u[u] = [result_emd, emd1, isParticioned, num_components]
        #-----------------ELECTION---------------------
        # First Criteria
        results_emd = [values[0] for values in results_u.values()]
        min_emd = min(results_emd)
        repeats = results_emd.count(min_emd)

        if repeats > 1:
            # Second Criteria
            results_u = {key: value for key, value in results_u.items() if value[0] == min_emd}
            emds_1 = [values[1] for values in results_u.values()]
            min_emds1 = min(emds_1)
            repeats = emds_1.count(min_emds1)
            if repeats > 1:
                # Third Criteria
                results_u = {key: value for key, value in results_u.items() if value[1] == min_emds1}
                booleans = [values[2] for values in results_u.values()]
                count_true = sum(booleans)
                if count_true >= 1:
                    # Pick the first one that is True
                    key = [key for key, value in results_u.items() if value[2] == True][0]
                else:
                    # Pick an edge randomly with min_emd
                    key = [key for key, value in results_u.items() if value[0] == min_emd][0]
            else: 
                key = [key for key, value in results_u.items() if value[1] == min_emds1][0]
        else:
            key = [key for key, value in results_u.items() if value[0] == min_emd][0]    
        if results_u[key][3] >= 3:
            b_break = True
        if isinstance(key, tuple):
            key = list(key)
        wp.remove(key)
        w_1.append(key)
        w_1l.append(key)
        results_u.clear()
    if len(w_1l) > 1:
        a.remove(w_1l[-1])
        a.remove(w_1l[-2])
        if isinstance(w_1l[-1], list) and isinstance(w_1l[-2], list):
            a.append(w_1l[-1] + w_1l[-2])

        elif isinstance(w_1l[-2], list):
            a.append([w_1l[-1]] + w_1l[-2])

        elif isinstance(w_1l[-1], list):
            a.append(w_1l[-1] + [w_1l[-2]])
        else:
            a.append([w_1l[-1], w_1l[-2]])
        candidates_bipartition = bipartition_system(
            df_tpm, a, initial_state, candidates_bipartition, node_state
        )
    return candidates_bipartition

def set_to_binary_1(set: list, present_label_len: int, future_label_len: int):
    """
    Convierte un conjunto de elementos en representaciones binarias para los estados presentes 
    y futuros basados en la longitud de las aristas proporcionadas.

    Args:
        set (list): Lista de elementos a convertir en binario. Los elementos pueden ser cadenas 
                    o listas de cadenas que contengan aristas.
        present_label_len (int): Longitud de la representación binaria para el estado presente.
        future_label_len (int): Longitud de la representación binaria para el estado futuro.

    Returns:
        list: Una lista que contiene dos cadenas binarias: la primera para el estado presente 
              y la segunda para el estado futuro.

    Example:
        >>> binary_states = set_to_binary_1(['aA', 'bA', 'bC'], 3, 3)
        >>> print(binary_states)  # Output: ['110', '101']
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

def set_to_binary_2(global_a: list, subset: list): 
    """
    Convierte un subconjunto de elementos en representaciones binarias para los estados presentes 
    y futuros, basándose en un conjunto global. Esta función se usa para recorrer listas de aristas

    Args:
        global_a (list): Lista que contiene todas las aristas del sistema completo
        subset (list): Lista de aristas a convertir en binario.

    Returns:
        list: Una lista que contiene dos cadenas binarias: la primera para el estado presente, 
              la segunda para el estado futuro, y una lista de carácteres que contienen todos los
              estados futuros presentes en subset.

    Example:
        >>> binary_states = set_to_binary_2(['aA', 'bA', 'aB', 'bB'], ['aA', 'bA'], 'aB'])
        >>> print(binary_states)  # Output: ['11', '11', ['A','B']]
    """
    abc = string.ascii_lowercase
    abc_upper = string.ascii_uppercase
    positions_to_keep_present = []
    positions_to_keep_future = []
    re_order_subset = []
    subset_t = []
    subset_t1 = []
    non_duplicated_group_t = []
    non_duplicated_group_t1 = []
    non_duplicated_subset_t = []
    non_duplicated_subset_t1 = []
    
    group_t = [elem[0] for elem in global_a]
    group_t1 = [elem[1] for elem in global_a]
    for elem in group_t:
        if elem not in non_duplicated_group_t:
            non_duplicated_group_t.append(elem)
    for elem in group_t1:
        if elem not in non_duplicated_group_t1:
            non_duplicated_group_t1.append(elem)
    
    indice_t = ord(max(non_duplicated_group_t)) - ord('a')
    
    indice_t_1 = ord(max(non_duplicated_group_t1)) - ord('A')
    
    # Re-order the subset (pop the elements that come into an array)
    for item in subset:
        if isinstance(item, list):
            for it in item:
                re_order_subset.append(it)
        else:
            re_order_subset.append(item)
    
    if len(re_order_subset) > 0:
        subset_t = [elem[0] for elem in re_order_subset]
        subset_t1 = [elem[1] for elem in re_order_subset]
        for elem in subset_t:
            if elem not in non_duplicated_subset_t:
                non_duplicated_subset_t.append(elem)
        for elem in subset_t1:
            if elem not in non_duplicated_subset_t1:
                non_duplicated_subset_t1.append(elem)
        
    binary_present = list(np.binary_repr(0, indice_t + 1))
    binary_future = list(np.binary_repr(0, indice_t_1 + 1))
    
    for i in range(len(binary_future)):
        binary_future[i] = '0'
        
    for j in range(len(binary_present)):
        binary_present[j] = '0'

    for elem in non_duplicated_subset_t:
        idx = abc.index(elem)
        binary_present[idx] = "1"
    
    for elem in non_duplicated_subset_t1:
        idx = abc_upper.index(elem)
        binary_future[idx] = "1"
                
    for k in non_duplicated_group_t:
        positions_to_keep_present.append(ord(k) - ord('a'))
        
    for l in non_duplicated_group_t1:
        positions_to_keep_future.append(ord(l) - ord('A'))
                
    binary_future = [binary_future[i] for i in positions_to_keep_future]
    binary_present = [binary_present[i] for i in positions_to_keep_present]
                
    return ["".join(binary_present), "".join(binary_future), non_duplicated_subset_t1]

def set_to_binary(global_a: list, element: str):
    """
    Convierte un elemento y un conjunto global en representaciones binarias para los estados 
    presentes y futuros, excluyendo el elemento especificado. Esta función se usa para representar
    el estado binario de una arista con respecto a todas las demás aristas

    Args:
        global_a (list): Lista de pares de elementos, donde el primer elemento representa 
                         el estado presente y el segundo el estado futuro.
        element (str): Un elemento en forma de cadena que se debe excluir del conjunto global.

    Returns:
        tuple: Una tupla que contiene dos cadenas binarias: la primera para el estado presente 
               y la segunda para el estado futuro de la arista.

    Example:
        >>> binary_states = set_to_binary(['aA', 'bA', 'aB', 'bB', 'aC', 'bC'], 'aC')
        >>> print(binary_states)  # Output: ('100', '001')
    """
    result = [s for s in global_a if s != element]
    present = []
    future  = []
    non_duplicated_present = []
    non_duplicated_future = []
    for el in result:
        present.append(el[0])
        future.append(el[1])
        
    for elem in present:
        if elem not in non_duplicated_present:
            non_duplicated_present.append(elem)
    for elem in future:
        if elem not in non_duplicated_future:
            non_duplicated_future.append(elem)
    present_str = []
    future_str = []
    present_digit, future_digit = list(element)
    for p in non_duplicated_present:
        if p == present_digit:
            present_str.append('0')
        else:
            present_str.append('1')
    for p in non_duplicated_future:
        if p == future_digit:
            future_str.append('0')
        else:
            future_str.append('1')
    
    present_str = ''.join(present_str)
    future_str = ''.join(future_str)
    return present_str, future_str

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

def build_a(present_subsystem: str, future_subsystem: str):
    """
    Construye una lista de aristas a partir de los subsistemas 
    presentes y futuros.

    Args:
        present_subsystem (str): Cadena que representa el estado presente, donde '1' indica 
                                 un estado activo.
        future_subsystem (str): Cadena que representa el estado futuro, donde '1' indica 
                                un estado activo.

    Returns:
        list: Lista de todas las posibles aristas, expresados en formato
              'xY', donde 'x' es la letra correspondiente al estado activo presente y 'Y' es la letra 
              correspondiente al estado activo futuro. Este string representa la conexión entre los dos
              nodos.

    Example:
        >>> a = build_a('110', '011')
        >>> print(a)  # Output: ['aB', 'bB', 'aC', 'bC']
    """
    a = []
    abc = string.ascii_lowercase
    abc_upper = string.ascii_uppercase
    for idx, bit in enumerate(present_subsystem):
        if bit == "1":
            for idx_2, bit_2 in enumerate(future_subsystem):
                if bit_2 == "1":
                    a.append(f"{abc[idx]}{abc_upper[idx_2]}")
    return a

def create_adjacency_matrix(subset: list):
    """
    Crea una matriz de adyacencia a partir de un subconjunto de aristas a desconectar.

    Args:
        subset (list): Lista de aristas que se utilizarán para modificar la matriz de 
                       adyacencia, desconectando cada elemento.

    Returns:
        np.ndarray: Matriz de adyacencia donde los elementos están marcados como '1' si 
                    hay una conexión y '0' si no hay conexión.

    Example:
        >>> adjacency_matrix = create_adjacency_matrix(['aB', 'bB', 'aC', 'bC'])
        >>> print(adjacency_matrix)  # Output: Matriz de adyacencia resultante
    """
    adjacency_matrix = global_adjacency_matrix.copy()
    
    for elem in subset:
        if isinstance(elem,list):
            for el in elem:
                present_el, future_el, _ = set_to_binary_2(global_a, [el])
                idx_row = 0
                idx_col = 0
                for row in adjacency_matrix:
                    if present_el[idx_row] == '1':
                        for _ in row:
                            if future_el[idx_col] == '1':
                                adjacency_matrix[idx_row, idx_col] = 0
                            idx_col += 1
                        idx_col = 0
                    idx_row += 1
        else:
            present_elem, future_elem, _ = set_to_binary_2(global_a, [elem])
            idx_row = 0
            idx_col = 0
            for row in adjacency_matrix:
                if present_elem[idx_row] == '1':
                    for _ in row:
                        if future_elem[idx_col] == '1':
                            adjacency_matrix[idx_row, idx_col] = 0
                        idx_col += 1
                    idx_col = 0
                idx_row += 1
    return adjacency_matrix

def count_bipartite_graph_components(adjacency_matrix):
    """
    Cuenta los componentes de un grafo bipartito representado por una matriz de adyacencia. Representa si el grafo
    se ha desconectado y en cuántas partes.

    Args:
        adjacency_matrix (np.ndarray): Matriz de adyacencia del grafo bipartito.

    Returns:
        tuple: Un tuple que contiene:
            - int: Número de componentes bipartitos.
            - list: Lista de componentes resultantes, donde cada componente es un diccionario con nodos de fila y columna.

    Example:
        >>> adjacency_matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        >>> num_components, components = count_bipartite_graph_components(adjacency_matrix)
        >>> print(num_components)  # Output: Número de componentes
        >>> print(components)  # Output: Qué nodos representan cada componente.
    """
    rows, cols = adjacency_matrix.shape
    visited_rows = np.zeros(rows, dtype=bool)
    visited_cols = np.zeros(cols, dtype=bool)
    
    components = []
    def depth_first_search(start_row, is_row_start):
        component = {'row_nodes': set(), 'col_nodes': set()}
        
        # Función interna de búsqueda recursiva
        def explore(current, is_current_row):
            if is_current_row:
                visited_rows[current] = True
                component['row_nodes'].add(current)
                
                # Explorar columnas conectadas
                for col in range(cols):
                    if adjacency_matrix[current, col] and not visited_cols[col]:
                        visited_cols[col] = True
                        component['col_nodes'].add(col)
                        # Explorar las filas conectadas a esta columna
                        for row in range(rows):
                            if adjacency_matrix[row, col] and not visited_rows[row]:
                                explore(row, True)
            else:
                visited_cols[current] = True
                component['col_nodes'].add(current)
                
                # Explorar filas conectadas
                for row in range(rows):
                    if adjacency_matrix[row, current] and not visited_rows[row]:
                        visited_rows[row] = True
                        component['row_nodes'].add(row)
                        # Explorar las columnas conectadas a esta fila
                        for col in range(cols):
                            if adjacency_matrix[row, col] and not visited_cols[col]:
                                explore(col, False)
        
        # Iniciar la exploración desde el nodo inicial
        if is_row_start:
            visited_rows[start_row] = True
            component['row_nodes'].add(start_row)
            
            # Explorar columnas conectadas
            for col in range(cols):
                if adjacency_matrix[start_row, col] and not visited_cols[col]:
                    visited_cols[col] = True
                    component['col_nodes'].add(col)
                    # Explorar las filas conectadas a esta columna
                    for row in range(rows):
                        if adjacency_matrix[row, col] and not visited_rows[row]:
                            explore(row, True)
        
        return component
    
    # Encontrar todos los componentes
    for start_row in range(rows):
        if not visited_rows[start_row]:
            current_component = depth_first_search(start_row, True)
            components.append(current_component)
    
    # Completar componentes no explorados por las filas
    for start_col in range(cols):
        if not visited_cols[start_col]:
            current_component = depth_first_search(start_col, False)
            components.append(current_component)
    return len(components), components

# Estrategia 2: Primer excel
def main_proof_cases():
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
    inicio = time.perf_counter()

    print(f"{initial_state=}, {candidate_system=}, {present_subsystem=}, {future_subsystem=}")
    df_tpm = apply_background(tensor_flow, initial_state, candidate_system)
    
    v = build_v(present_subsystem, future_subsystem)
    a = build_a(present_subsystem, future_subsystem)
    global global_a
    global_a = a.copy()
    global marginalized_tpm
    marginalized_tpm = {}
    global global_adjacency_matrix 
    present_global, future_global, _ = set_to_binary_2(a, a)
    global_adjacency_matrix = np.ones((len(present_global), len(future_global)))
    
    global viewed_tpm
    viewed_tpm = {}
    
    present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))

    node_states = get_first_matrices_node_state(df_tpm)

    result_df = first_marginalize_node_states(present, node_states, sorted(node_states.keys()))
    result_df = marginalize_cols(result_df, future)
    node_states = get_matrices_node_state(result_df, future)
    
    candidates_bipartition = {}
    candidates_bipartition['0'] = [[], 10000]
    candidate_bipartitions = bipartition_system(
        result_df.copy(), a.copy(), initial_state, candidates_bipartition, node_states
    )
    print(f"{candidate_bipartitions=}")
    fin = time.perf_counter()
    print("Tiempo=")
    print(fin-inicio)

# caso de prueba red 10
def main():
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
    a = build_a(present_subsystem, future_subsystem)

    global global_a
    global_a = a.copy()
    global marginalized_tpm
    marginalized_tpm = {}
    present_global, future_global, _ = set_to_binary_2(a, a)
    global global_adjacency_matrix 
    global_adjacency_matrix = np.ones((len(present_global), len(future_global)))

    present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))

    node_states = get_first_matrices_node_state(df_tpm)

    result_df = first_marginalize_node_states(present, node_states, sorted(node_states.keys()))
    result_df = marginalize_cols(result_df, future)
    node_states = get_matrices_node_state(result_df, future)
    
    candidates_bipartition = {}
    candidates_bipartition['0'] = [[], 10000]
    candidate_bipartitions = bipartition_system(
        result_df.copy(), a.copy(), initial_state, candidates_bipartition, node_states
    )
    print(f"{initial_state=}, {candidate_system=}, {present_subsystem=}, {future_subsystem=}")
    print(f"{candidate_bipartitions=}")
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
    a = build_a(present_subsystem, future_subsystem)

    global global_a
    global_a = a.copy()
    global marginalized_tpm
    marginalized_tpm = {}
    present_global, future_global, _ = set_to_binary_2(a, a)
    global global_adjacency_matrix 
    global_adjacency_matrix = np.ones((len(present_global), len(future_global)))

    present, future = set_to_binary_1(v, len(df_tpm.index[0]), len(df_tpm.columns[0]))

    node_states = get_first_matrices_node_state(df_tpm)

    result_df = first_marginalize_node_states(present, node_states, sorted(node_states.keys()))
    result_df = marginalize_cols(result_df, future)
    node_states = get_matrices_node_state(result_df, future)
    
    candidates_bipartition = {}
    candidates_bipartition['0'] = [[], 10000]
    inicio = time.perf_counter()
    candidate_bipartitions = bipartition_system(
        result_df.copy(), a.copy(), initial_state, candidates_bipartition, node_states
    )
    fin = time.perf_counter()

    return [candidate_bipartitions['0'][0], candidate_bipartitions['0'][1], fin-inicio]
# main()