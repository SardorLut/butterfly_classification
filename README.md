# Butterfly Classification

Система классификации бабочек на основе PyTorch Lightning с поддержкой 75 классов различных видов бабочек. Проект включает полный пайплайн от обучения до продакшен-готовых моделей в форматах ONNX и TensorRT.

## Описание проекта

Классификатор способен определить вид бабочки по фотографии из 75 возможных классов. Система построена с использованием современных практик MLOps и готова к развертыванию в продакшене.

**Основные возможности:**
- Обучение моделей на базе предобученных ResNet архитектур(Resnet[18,50,101])
- Автоматическое логирование экспериментов в MLflow
- Скачивание данных через скрипт
- Экспорт моделей в ONNX и TensorRT для оптимизированного инференса
- Настраиваемые конфигурации через Hydra
- Качественный код с pre-commit хуками

**Технические характеристики:**
- 75 классов бабочек
- ~6,500 тренировочных изображений
- ~2,500 тестовых изображений
- Точность валидации: 85-90%

## Структура проекта

```
butterfly_classification/
├── butterfly_classification/           # Основной Python пакет
│   ├── train_pipeline/                # Пайплайн обучения
│   ├── inference/                     # Пайплайн инференса
│   └── model_conversion_pipeline/     # Конвертация моделей
├── configs/                           # Hydra конфигурации
├── data/                             # Данные (управляется DVC)
├── outputs/                          # Результаты обучения
├── plots/                           # Графики и визуализации
├── Makefile                         # Автоматизация команд
├── pyproject.toml                   # Poetry зависимости
└── README.md                        # Документация
```

## Setup

### 1. Клонирование репозитория

```bash

```

### 2. Установка зависимостей

```bash
# Установка Poetry (если не установлен)
curl -sSL https://install.python-poetry.org | python3 -

# Установка зависимостей проекта
make install
```

### 3. Настройка pre-commit

```bash
make setup-precommit
```

### 4. Загрузка данных

```bash
make download-dataset
```

### 5. Запуск MLflow сервера

```bash
# В отдельном терминале
make mlflow-server
```

## Train

### Быстрый старт обучения
запустить mlflow сервер через make run-mlflow-server
```bash
# Обучение с дефолтными параметрами
make train

# Обучение с кастомными параметрами
make train-custom model.model_name=resnet101 trainer.max_epochs=50
```

### Доступные команды обучения
доступны в конфиге
```bash

# Обучение с конкретной конфигурацией
python -m butterfly_classification.train_pipeline.train \
    model.model_name=resnet50 \
    trainer.max_epochs=30 \
    datamodule.batch_size=32
```

### Мониторинг обучения

Откройте MLflow UI в браузере: http://127.0.0.1:8080

## Production preparation

### Конвертация в ONNX

```bash
# Конвертация лучшей модели в ONNX
make convert-onnx

# Конвертация конкретной модели
make convert-onnx CHECKPOINT_PATH=outputs/checkpoints/model.ckpt
```

### Конвертация в TensorRT

```bash
# Конвертация ONNX модели в TensorRT
make convert-tensorrt

# Конвертация с кастомными параметрами
python convert_to_tensorrt.py \
    --onnx_path outputs/models/model.onnx \
    --engine_path outputs/models/model.engine \
    --precision fp16
```

### Артефакты поставки

После обучения и конвертации у вас будут следующие артефакты:

```
outputs/
├── checkpoints/
│   └── best_model.ckpt              # Исходная Lightning модель
├── models/
│   ├── model.onnx                   # ONNX модель
│   └── model.engine                 # TensorRT модель
└── configs/
    └── model_config.yaml            # Конфигурация модели
```

Для продакшена необходимы:
- `model.onnx` или `model.engine` - модель для инференса
- `model_config.yaml` - конфигурация модели
- `butterfly_classification/inference/` - код для инференса

## Infer

### Предсказания для тестового набора

```bash
# Инференс с моделью
make predict
```

### Предсказание для одного изображения

```bash
# С Lightning моделью
make predict-single IMAGE_PATH=path/to/butterfly.jpg

# С ONNX моделью
python -m butterfly_classification.inference.predict \
    data.single_image=path/to/butterfly.jpg \
    model.checkpoint_path=outputs/models/model.onnx
```

### Формат входных данных

**Для одного изображения:**
- Поддерживаемые форматы: JPG, PNG
- Рекомендуемое разрешение: 224x224 пикселей
- Канал: RGB

**Для батча изображений:**
Создайте CSV файл с колонкой `filename`:
```csv
filename
butterfly1.jpg
butterfly2.jpg
butterfly3.jpg
```

### Формат выходных данных

Система создает следующие файлы:

1. **predictions.csv** - детальные результаты:
```csv
filename,predicted_class,confidence
butterfly1.jpg,MONARCH,0.98
butterfly2.jpg,SWALLOWTAIL,0.87
```

2. **submission.csv** - файл для отправки:
```csv
filename,label
butterfly1.jpg,MONARCH
butterfly2.jpg,SWALLOWTAIL
```

3. **probabilities.csv** - все вероятности по классам

## Доступные команды Make

```bash
# Установка и настройка
make install              # Установка зависимостей
make pre-commit-install     # Настройка pre-commit
make download-data        # Загрузка данных через DVC

# Обучение
make train               # Обучение с дефолтными параметрами

# Конвертация моделей
make convert-onnx        # Конвертация в ONNX
make convert-tensorrt    # Конвертация в TensorRT

# Инференс
make run-inference-pipeline             # Предсказания (Lightning)make predict-tensorrt

# Утилиты
make mlflow-server       # Запуск MLflow сервера
make clean              # Очистка временных файлов
make format             # Запуск линтеров
```

## Конфигурация

Все параметры настраиваются через Hydra конфигурации в папке `configs/`:

- `train_config.yaml` - параметры обучения
- `predict_config.yaml` - параметры инференса
- `convert_onnx_config.yaml` - параметры конвертации ONNX
- `convert_tensorrt_config.yaml` - параметры конвертации TensorRT

Пример изменения параметров:
```bash
python -m butterfly_classification.train_pipeline.train \
    model.model_name=resnet101 \
    trainer.max_epochs=50 \
    datamodule.batch_size=64 \
    optimizer.learning_rate=1e-4
```

## Требования к системе

**Минимальные требования:**
- Python 3.8+
- 8GB RAM
- 10GB свободного места

**Рекомендуемые требования:**
- GPU с поддержкой CUDA
- 16GB RAM
- 50GB свободного места

**Для TensorRT:**
- NVIDIA GPU
- CUDA 11.0+
- TensorRT 8.0+

## Лицензия

MIT License
