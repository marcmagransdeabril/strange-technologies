# Tecnologías Extrañas

*Apuestas asimétricas para ingenieros curiosos*

Marc Magrans de Abril

📖 [Leer en línea](https://marcmagransdeabril.github.io/strange-technologies/book/)

## Contenido

- Cifrado Homomórfico
- Análisis Topológico de Datos

## Código

Los ejemplos de código ejecutable están organizados por capítulo:

```
code/
├── i18n/
├── cifrado-homomorfico/
└── tda/
tests/
├── cifrado-homomorfico/code/
└── tda/code/
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
