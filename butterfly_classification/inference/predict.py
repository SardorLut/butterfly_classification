#!/usr/bin/env python3
"""
Скрипт для предсказаний на тестовом наборе данных с использованием Hydra
Поддерживает модели в форматах: .ckpt (PyTorch Lightning), .onnx, .engine/.trt (TensorRT)
"""

import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from butterfly_classification.train_pipeline.dataset import get_transforms
from butterfly_classification.train_pipeline.model_loader import load_model

# Игнорируем предупреждения
warnings.filterwarnings("ignore")


def predict_single_image(model_wrapper, image_path, transform, class_names):
    """Предсказание для одного изображения с универсальной моделью."""

    # Загружаем и преобразуем изображение
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Получаем предсказание через универсальный интерфейс
    outputs = model_wrapper.predict(image_tensor)

    # Конвертируем в тензор для дальнейшей обработки
    if isinstance(outputs, np.ndarray):
        outputs = torch.from_numpy(outputs)

    # Вычисляем softmax и получаем предсказание
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_class_idx].item()

    predicted_class = class_names[predicted_class_idx]

    return predicted_class, confidence, probabilities[0].cpu().numpy()


def predict_test_dataset(cfg: DictConfig):
    """Предсказания для всего тестового набора."""

    if cfg.logging.verbose:
        print("=" * 50)
        print("🦋 BUTTERFLY CLASSIFICATION PREDICTIONS 🦋")
        print("=" * 50)
        print(f"Конфигурация:\n{OmegaConf.to_yaml(cfg)}")

    # Создаем выходную директорию
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем модель через универсальный загрузчик
    if cfg.logging.verbose:
        print(" Загружаем модель из {}".format(cfg.model.checkpoint_path))

    try:
        model_wrapper = load_model(cfg.model.checkpoint_path)

        # Выводим информацию о модели
        model_info = model_wrapper.get_model_info()
        if cfg.logging.verbose:
            print("Тип модели: {}".format(model_info["model_type"]))
            print("Устройство: {}".format(model_info["device"]))
            print("Количество классов: {}".format(model_info["num_classes"]))

            if model_info["model_type"] == "onnx":
                print("ONNX провайдеры: {}".format(model_info["providers"]))
            elif model_info["model_type"] == "tensorrt":
                print(
                    "TensorRT максимальный batch size: {}".format(
                        model_info.get("max_batch_size", "не указан")
                    )
                )

    except Exception as e:
        print("Ошибка загрузки модели: {}".format(e))
        return None

    # Получаем названия классов
    class_names = model_wrapper.class_names
    if not class_names:
        # Если названия классов не сохранены в модели, используем дефолтные
        class_names = [f"Class_{i}" for i in range(model_info["num_classes"] or 75)]

    # Получаем трансформации (только для валидации/теста)
    transform = get_transforms(train=False)

    # Загружаем список тестовых файлов
    test_df = pd.read_csv(cfg.data.test_csv)
    if cfg.logging.verbose:
        print(f"Количество тестовых изображений: {len(test_df)}")

    # Готовим результаты
    predictions = []
    confidences = []
    all_probabilities = []

    # Проходим по всем тестовым изображениям
    if cfg.logging.verbose:
        print("Делаем предсказания...")

    iterator = (
        tqdm(test_df.iterrows(), total=len(test_df))
        if cfg.processing.show_progress
        else test_df.iterrows()
    )

    for idx, row in iterator:
        filename = row["filename"]
        image_path = Path(cfg.data.test_images_dir) / filename

        if image_path.exists():
            try:
                pred_class, confidence, probs = predict_single_image(
                    model_wrapper, image_path, transform, class_names
                )
                predictions.append(pred_class)
                confidences.append(confidence)
                all_probabilities.append(probs)
            except Exception as e:
                if cfg.logging.verbose:
                    print(f"Ошибка обработки {filename}: {e}")
                predictions.append("UNKNOWN")
                confidences.append(0.0)
                all_probabilities.append(np.zeros(len(class_names)))
        else:
            if cfg.logging.verbose:
                print(f"Файл не найден: {image_path}")
            predictions.append("UNKNOWN")
            confidences.append(0.0)
            all_probabilities.append(np.zeros(len(class_names)))

    # Создаем DataFrame с результатами
    results_df = test_df.copy()
    results_df["predicted_class"] = predictions
    results_df["confidence"] = confidences

    # Сохраняем результаты согласно конфигурации
    saved_files = []

    if cfg.output.create_detailed_results:
        results_path = output_dir / "predictions.csv"
        results_df.to_csv(results_path, index=False)
        saved_files.append(str(results_path))
        if cfg.logging.verbose:
            print(f"Детальные результаты сохранены: {results_path}")

    if cfg.output.create_submission:
        submission_df = pd.DataFrame(
            {"filename": test_df["filename"], "label": predictions}
        )
        submission_path = output_dir / "submission.csv"
        submission_df.to_csv(submission_path, index=False)
        saved_files.append(str(submission_path))
        if cfg.logging.verbose:
            print(f"Файл для отправки: {submission_path}")

    if cfg.output.create_probabilities:
        probs_df = pd.DataFrame(all_probabilities, columns=class_names)
        probs_df["filename"] = test_df["filename"]
        probs_path = output_dir / "probabilities.csv"
        probs_df.to_csv(probs_path, index=False)
        saved_files.append(str(probs_path))
        if cfg.logging.verbose:
            print(f"Детальные вероятности: {probs_path}")

    # Статистика предсказаний
    if cfg.logging.show_statistics and cfg.logging.verbose:
        print(f"\nСтатистика предсказаний (топ-{cfg.logging.top_k_stats}):")
        pred_stats = pd.Series(predictions).value_counts().head(cfg.logging.top_k_stats)
        for class_name, count in pred_stats.items():
            percentage = (count / len(predictions)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")

    if cfg.logging.verbose:
        print("\nПредсказания завершены!")
        print(f"Все файлы сохранены в: {output_dir}")
        for file_path in saved_files:
            print(f"  - {file_path}")

        # Информация о производительности
        model_type = model_wrapper.model_type
        if model_type == "onnx":
            print("Модель ONNX обеспечивает оптимизированную производительность")
        elif model_type == "tensorrt":
            print("Модель TensorRT обеспечивает максимальную производительность на GPU")
        else:
            print("Использована PyTorch Lightning модель")

    return results_df


def predict_single_image_mode(cfg: DictConfig):
    """Предсказание для одного изображения."""

    if cfg.logging.verbose:
        print("Предсказание для одного изображения...")

    # Загружаем модель
    try:
        model_wrapper = load_model(cfg.model.checkpoint_path)

        # Выводим информацию о модели
        model_info = model_wrapper.get_model_info()
        if cfg.logging.verbose:
            print(f"Тип модели: {model_info['model_type']}")
            print(f"Устройство: {model_info['device']}")

    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None, None

    # Получаем трансформации
    transform = get_transforms(train=False)

    # Получаем названия классов
    class_names = model_wrapper.class_names
    if not class_names:
        class_names = [f"Class_{i}" for i in range(75)]

    # Делаем предсказание
    pred_class, confidence, _ = predict_single_image(
        model_wrapper, cfg.data.single_image, transform, class_names
    )

    if cfg.logging.verbose:
        print(f"Изображение: {cfg.data.single_image}")
        print(f"Предсказанный класс: {pred_class}")
        print(f"Уверенность: {confidence:.3f}")
        print(f"Устройство: {model_wrapper.device}")
        print(f"Формат модели: {model_wrapper.model_type}")

    return pred_class, confidence


@hydra.main(
    version_base=None, config_path="../../configs", config_name="predict_config"
)
def main(cfg: DictConfig) -> None:
    """Основная функция предсказаний."""

    # Проверяем, что checkpoint существует
    if not os.path.exists(cfg.model.checkpoint_path):
        print(f"Файл модели не найден: {cfg.model.checkpoint_path}")
        return

    # Определяем формат модели
    model_extension = Path(cfg.model.checkpoint_path).suffix.lower()
    supported_formats = [".ckpt", ".onnx", ".engine", ".trt", ".pth", ".pt"]

    if model_extension not in supported_formats:
        print(f"Неподдерживаемый формат модели: {model_extension}")
        print(f"   Поддерживаемые форматы: {', '.join(supported_formats)}")
        return

    if cfg.logging.verbose:
        print(f"Обнаружен формат модели: {model_extension}")

    if cfg.data.single_image:
        # Режим предсказания для одного изображения
        if not os.path.exists(cfg.data.single_image):
            print(f"Изображение не найдено: {cfg.data.single_image}")
            return
        predict_single_image_mode(cfg)
    else:
        # Режим предсказания для тестового набора
        if not os.path.exists(cfg.data.test_csv):
            print(f"CSV файл не найден: {cfg.data.test_csv}")
            return
        if not os.path.exists(cfg.data.test_images_dir):
            print(f"Директория с изображениями не найдена: {cfg.data.test_images_dir}")
            return
        predict_test_dataset(cfg)


if __name__ == "__main__":
    main()
