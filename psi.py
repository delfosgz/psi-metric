import pandas as pd
import numpy as np

def calculate_psi_column(expected_array, actual_array, bucket_type='bins', buckets=10, epsilon=1e-4):
    """
    Calcula el PSI para una sola columna (numérica o categórica).
    """
    
    # --- Lógica para Variables Categóricas ---
    if bucket_type == 'category':
        # Calcular frecuencias relativas
        expected_percents = pd.Series(expected_array).value_counts(normalize=True)
        actual_percents = pd.Series(actual_array).value_counts(normalize=True)
        
        # Alinear índices (para que coincidan las categorías en ambos lados)
        # Esto maneja categorías nuevas que no existían en el entrenamiento o viceversa
        combined_index = expected_percents.index.union(actual_percents.index)
        expected_percents = expected_percents.reindex(combined_index, fill_value=0)
        actual_percents = actual_percents.reindex(combined_index, fill_value=0)

    # --- Lógica para Variables Numéricas ---
    else:
        # 1. Crear los cortes (bins) usando el dataset de ENTRENAMIENTO (expected)
        # Se usa qcut para tratar de tener bins de igual tamaño poblacional
        try:
            _, bins = pd.qcut(expected_array, q=buckets, retbins=True, duplicates='drop')
        except ValueError:
            # Si hay muy pocos valores únicos, se trata como categórica
            return calculate_psi_column(expected_array, actual_array, bucket_type='category')
        
        # Ajustar los bordes para incluir máximos y mínimos extremos del nuevo dataset
        bins[0] = min(bins[0], actual_array.min()) - 0.001
        bins[-1] = max(bins[-1], actual_array.max()) + 0.001
        
        # 2. Aplicar esos MISMOS cortes al dataset NUEVO y al VIEJO
        expected_binned = pd.cut(expected_array, bins=bins, include_lowest=True)
        actual_binned = pd.cut(actual_array, bins=bins, include_lowest=True)
        
        # 3. Calcular porcentajes
        expected_percents = expected_binned.value_counts(normalize=True).sort_index()
        actual_percents = actual_binned.value_counts(normalize=True).sort_index()

    # --- Cálculo Matemático del PSI ---
    # Se añade epsilon para evitar división por cero o logaritmo de cero
    expected_percents = expected_percents + epsilon
    actual_percents = actual_percents + epsilon
    
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = psi_values.sum()
    
    return psi

def get_psi_report(df_train, df_score):
    """
    Recorre todas las columnas comunes y genera un reporte de PSI.
    """
    psi_list = []
    
    # Identificar columnas comunes
    common_cols = list(set(df_train.columns) & set(df_score.columns))
    
    for col in common_cols:
        # Detectar si es numérica o categórica
        is_numeric = pd.api.types.is_numeric_dtype(df_train[col])
        bucket_type = 'bins' if is_numeric else 'category'
        
        # Calcular PSI
        psi_val = calculate_psi_column(df_train[col], df_score[col], bucket_type=bucket_type)
        
        # Definir semáforo
        if psi_val < 0.1:
            status = 'Verde (Estable)'
        elif psi_val < 0.25:
            status = 'Amarillo (Alerta)'
        else:
            status = 'Rojo (Cambio Crítico)'
            
        psi_list.append({
            'Variable': col,
            'Tipo': 'Numérica' if is_numeric else 'Categórica',
            'PSI': round(psi_val, 4),
            'Estatus': status
        })
    
    # Crear DataFrame de resultados ordenado por PSI descendente
    psi_report = pd.DataFrame(psi_list).sort_values(by='PSI', ascending=False)
    return psi_report