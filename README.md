# LeanStats - Intervalos de Confianza

Este repositorio contiene análisis estadísticos relacionados con el cálculo de intervalos de confianza para proporciones utilizando diferentes métodos y librerías de Python.

## Contenido

- `intervalo_confianza.ipynb`: Notebook principal que compara diferentes métodos para calcular:
  - Tamaño de muestra necesario para intervalos de confianza
  - Margen de error para proporciones
  - Comparación entre scipy.stats, statsmodels, y otros enfoques

## Métodos Implementados

### Cálculo de Tamaño de Muestra
1. **Método manual con scipy.stats**: Usando la fórmula clásica con distribución normal
2. **statsmodels**: Usando `samplesize_confint_proportion`

### Cálculo de Margen de Error
1. **scipy.stats con distribución normal**: Método estándar
2. **statsmodels**: Usando `proportion_confint`
3. **scipy.stats con t-Student**: Más preciso para muestras pequeñas
4. **numpy con percentiles**: Enfoque alternativo
5. **pingouin**: Librería especializada en estadística

## Requisitos

```bash
pip install scipy
pip install statsmodels
pip install numpy
pip install pingouin  # Opcional
```

## Uso

Abrir el notebook `intervalo_confianza.ipynb` en Jupyter Lab/Notebook o VS Code para ejecutar los análisis.

## Parámetros de Ejemplo

- Nivel de confianza: 95% y 98%
- Margen de error: 3%
- Proporción estimada: 0.5 (máximo margen de error)
- Tamaño de muestra de ejemplo: 232

## Autor

Proyecto de análisis estadístico para comparar diferentes enfoques de cálculo de intervalos de confianza.
