# PSI Calculator (Drift Detection) (AI Made) 📊

Módulo en Python para calcular el **Índice de Estabilidad Poblacional (PSI)** y detectar cambios drásticos en la distribución de datos (Data Drift) entre dos periodos.

## 🚦 Reglas de Interpretación

| Valor PSI | Estatus | Significado | Acción Recomendada |
| :--- | :--- | :--- | :--- |
| **< 0.10** | 🟢 **Verde** | Población Estable | Ninguna. El modelo es seguro. |
| **0.10 - 0.25** | 🟡 **Amarillo** | Cambio Moderado | Precaución. Revisar variables afectadas. |
| **> 0.25** | 🔴 **Rojo** | Cambio Crítico | **Alerta.** La población cambió. Reentrenar modelo. |

## 📋 Requisitos
* Python 3.x
* `pandas`
* `numpy`

## 🛠 Funciones Principales

### `get_psi_report(df_train, df_score)`
Compara todas las columnas comunes entre dos DataFrames y devuelve una tabla de resultados.
* **Lógica:** Detecta automáticamente si la variable es numérica (usa deciles fijos del train) o categórica.
* **Retorno:** DataFrame ordenado por PSI descendente.

### `calculate_psi_column(expected, actual)`
Calcula el valor escalar del PSI para una sola variable (array/serie).