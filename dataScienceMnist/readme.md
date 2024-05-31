# Readme

## Introducción

Este proyecto consiste en una evaluación de los modelos entrenados en el archivo Jupyter Notebook que tiene como nombre models_training. 
Se ha utilizado el dataset MNIST el cual contiene imágenes para el reconocimiento  de dígitos escritos a mano.


## Estructura del Código

El código está organizado en varios archivos para mejorar la modularidad.

- `constants.py`: Define las constantes utilizadas en todo el proyecto.
- `data_manager.py`: Contiene la clase DataManager, que se encarga de cargar y procesar los datos del conjunto de datos MNIST.
- `evaluation_logic.py`: Contiene la clase EvaluationLogic, que maneja la lógica principal del programa, incluida la interacción con el usuario y la ejecución de la evaluación del modelo.
- `model_evaluator.py`: Contiene la clase ModelEvaluator, que se encarga de evaluar el rendimiento del modelo.
- `model_manager.py`: Contiene la clase ModelManager, que gestiona la carga de modelos previamente entrenados.

La carpeta `trained_models` contiene los tres modelos previamente entrenados en el Jupyter Notebook. El nombre de los archivos se define en el archivo de configuración `constants.py`.

## Decisiones de Alto Nivel

### Estructura del Código

- **Modularidad**: Se ha organizado el código en módulos separados para cada componente funcional (carga de datos, lógica de evaluación, gestión de modelos, etc.), con la intención de facilitar la comprensión del proyecto.
- **Clases**: Se han utilizado clases para encapsular la funcionalidad relacionada y poder hacer uso de la reutilización del código.
- **Estándares de Python**: Se ha documentado todo el código para permitir su legibilidad.

### Librerías Externas

En el archivo requirements.txt se encuentran las librerías utilizadas junto con sus versiones.

