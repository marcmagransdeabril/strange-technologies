# Tecnologías Extrañas

*Apuestas asimétricas para ingenieros curiosos*

Hay tecnologías que la mayoría de ingenieros nunca encontrará en su trabajo diario. No porque sean inútiles, sino porque son difíciles, están en las fronteras entre disciplinas, o simplemente no han encontrado todavía su momento. Aprenderlas es como comprar una opción *call* fuera del dinero: el coste es pequeño — unas pocas tardes de estudio —, la pérdida máxima está acotada, pero si la tecnología despega, el retorno profesional no tiene techo. Para separar señal de ruido, este libro selecciona sus 24 tecnologías con un filtro doble: el Efecto Lindy — priorizando ideas con décadas de maduración teórica — y cuatro métricas empíricas que trazan el ciclo de adopción (publicaciones académicas, visibilidad en Hacker News, tracción en GitHub y financiación de startups). El resultado es un mapa de apuestas asimétricas: tecnologías extrañas, poderosas y probablemente más relevantes de lo que parecen.

*Marc Magrans de Abril*

📖 [Leer en línea](https://marcmagransdeabril.github.io/strange-technologies/book/)

## Contenido

### Introducción

- Tecnología

### Parte I — En Órbita

- CRDTs
- [Cifrado Homomórfico](https://marcmagransdeabril.github.io/strange-technologies/book/chapters/cifrado-homomorfico.es.html) [EN](https://marcmagransdeabril.github.io/strange-technologies/book/chapters/cifrado-homomorfico.en.html) [📄<sub>ES</sub>](https://github.com/marcmagransdeabril/strange-technologies/raw/main/book/chapters/cifrado-homomorfico.es.pdf) [📄<sub>EN</sub>](https://github.com/marcmagransdeabril/strange-technologies/raw/main/book/chapters/cifrado-homomorfico.en.pdf)
- Aprendizaje Profundo Geométrico
- Inferencia Causal
- Kernels de Tangente Neural
- Análisis del Panorama de Optimización
- Privacidad Diferencial
- Programación Probabilística
- Síntesis de Programas
- Teoría de Categorías Computacional
- Teoría Espectral de Grafos

### Parte II — En Rampa

- Criptografía de Retículos
- Pruebas de Conocimiento Cero
- Tolerancia a Fallos Bizantinos
- Transporte Óptimo
- Verificación Formal e IA
- Compresión Aprendida

### Parte III — En el Laboratorio

- Teoremas de Aproximación Universal
- Computación Analógica
- Recuperación de Señales Dispersas
- 👷 Análisis Topológico de Datos
- Estadística Algebraica
- Teoría de Juegos Algorítmica

## Código

Los ejemplos de código ejecutable están organizados por capítulo:

```
code/
├── cifrado-homomorfico/
│   ├── quick_start.py
│   └── bootstrapping.py
└── i18n/
```
```
tests/
├── conftest.py
└── cifrado-homomorfico/
    ├── conftest.py
    └── code/
        ├── test_quick_start.py
        └── test_bootstrapping.py
```

### Ejecución de tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q --tb=short
```

Algunos capítulos requieren dependencias opcionales pesadas (tenseal, openfhe).
Si no están instaladas, los tests correspondientes se saltan automáticamente.
Para instalarlas:

```bash
pip install tenseal    # ~500 MB, cifrado homomórfico
pip install openfhe    # bootstrapping CKKS
```
