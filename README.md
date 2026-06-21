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
- [Cifrado Homomórfico](https://marcmagransdeabril.github.io/strange-technologies/book/chapters/cifrado-homomorfico.es.html) [📄<sub>ES</sub>](https://github.com/marcmagransdeabril/strange-technologies/raw/main/book/chapters/cifrado-homomorfico.es.pdf) [📄<sub>EN</sub>](https://github.com/marcmagransdeabril/strange-technologies/raw/main/book/chapters/cifrado-homomorfico.en.pdf)
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
- [Análisis Topológico de Datos](https://marcmagransdeabril.github.io/strange-technologies/book/chapters/tda.es.html) [📄<sub>ES</sub>](https://github.com/marcmagransdeabril/strange-technologies/raw/main/book/chapters/tda.es.pdf) [📄<sub>EN</sub>](https://github.com/marcmagransdeabril/strange-technologies/raw/main/book/chapters/tda.en.pdf)
- Estadística Algebraica
- Teoría de Juegos Algorítmica
- Biología Sintética

## Código

Los ejemplos de código ejecutable están organizados por capítulo:

```
code/
├── i18n/                    # Módulo i18n y cadenas localizadas
├── cifrado-homomorfico/     # Scripts del capítulo de cifrado homomórfico
└── tda/                     # Scripts del capítulo de análisis topológico
tests/
├── cifrado-homomorfico/code/  # Tests unitarios
└── tda/code/                  # Tests unitarios
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
