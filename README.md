# Optimización Metaheurística con Bat Algorithm (BA) - Grupo 5
**Autor:** Jorge Giraldo Rincón

## Contenido del repositorio
- `src/ba_optimizer.py`: Implementación del Bat Algorithm (Yang, 2010), búsqueda aleatoria y contador estricto de 3,000 evaluaciones.
- `src/problems_f5_e3.py`: Función F5 (Schaffer N.2 desplazada con `s = (23.73, -26.60)`), traza de la iteración 1 con la matriz `U` y modelo del filtro Butterworth Sallen-Key de 4.º orden con series comerciales `E24`/`E12` (E3).
- `src/run_experiments.py`: Protocolo de 30 corridas independientes (semillas 1 a 30), prueba estadística de Wilcoxon (`alfa = 0.05`) y análisis de sensibilidad de `r0`.
- `src/validate_e3.py`: Validaciones V1 (respuesta ideal en frecuencia), V2 (efecto GBW de Op-Amps reales y netlist SPICE) y V3 (Monte Carlo de 500 circuitos con tolerancias `E24 ±5%` y `E12 ±10%`).
- `results/`: Figuras de convergencia, sensibilidad, Bode, histogramas Monte Carlo y netlist `.cir`.

## Reproducción
1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt