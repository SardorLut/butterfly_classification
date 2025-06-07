import logging
from pathlib import Path

import hydra
import torch
import torchvision.models as models
from omegaconf import DictConfig

from butterfly_classification.model_conversion_pipeline.utils import (
    convert_to_onnx,
    convert_to_tensorrt,
)
from butterfly_classification.train_pipeline.model import ButterflyClassifier

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str, num_classes: int, device: str = "cpu"
) -> torch.nn.Module:
    """
    Загружает модель из чекпоинта.

    Args:
        checkpoint_path: Путь к чекпоинту модели
        num_classes: Количество классов
        device: Устройство для загрузки модели

    Returns:
        torch.nn.Module: Загруженная модель
    """
    try:
        # Попытаемся определить архитектуру модели из чекпоинта
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Если чекпоинт содержит полную модель
        if "model" in checkpoint:
            model = checkpoint["model"]
        # Если чекпоинт содержит state_dict (Lightning checkpoints)
        elif "state_dict" in checkpoint:
            # Получаем гиперпараметры из чекпоинта или используем значения по умолчанию
            hyper_parameters = checkpoint.get("hyper_parameters", {})
            model_name = hyper_parameters.get("model_name", "resnet50")

            # Создаем Lightning модель
            lightning_model = ButterflyClassifier(
                num_classes=num_classes,
                model_name=model_name,
                pretrained=False,  # Мы загружаем веса из чекпоинта
            )

            # Загружаем состояние модели
            lightning_model.load_state_dict(checkpoint["state_dict"])

            # Переводим модель в режим eval перед извлечением
            lightning_model.eval()

            # Извлекаем только PyTorch модель (без Lightning обертки)
            model = lightning_model.model

            # Убеждаемся, что модель в режиме eval
            model.eval()

            # Отключаем градиенты для всех параметров
            for param in model.parameters():
                param.requires_grad = False

        else:
            # Если чекпоинт содержит только веса модели напрямую
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()  # Важно: модель должна быть в eval режиме

        logger.info(f"Модель успешно загружена из {checkpoint_path}")
        logger.info(f"Архитектура модели: {model.__class__.__name__}")

        return model

    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise


@hydra.main(
    version_base=None, config_path="../../configs", config_name="model_conversion"
)
def main(config: DictConfig):
    """
    Основная функция пайплайна конвертации модели.

    Args:
        config: Конфигурация из YAML файла
    """
    logger.info("Запуск пайплайна конвертации модели")

    try:
        # Проверяем существование файла чекпоинта
        checkpoint_path = config.model.checkpoint_path
        if not Path(checkpoint_path).exists():
            logger.error(f"Файл чекпоинта не найден: {checkpoint_path}")
            return

        # Загружаем модель
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используем устройство: {device}")

        model = load_model(
            checkpoint_path=checkpoint_path,
            num_classes=config.model.num_classes,
            device=device,
        )

        # Конвертация в ONNX
        if config.conversion.convert_to_onnx:
            logger.info("Начинаем конвертацию в ONNX...")

            onnx_success = convert_to_onnx(
                model=model,
                input_shape=config.model.input_shape,
                output_path=config.onnx.output_path,
                opset_version=config.onnx.opset_version,
                dynamic_axes=config.onnx.dynamic_axes,
            )

            if onnx_success:
                logger.info("Конвертация в ONNX завершена успешно")
            else:
                logger.error("Ошибка при конвертации в ONNX")
                return
        else:
            logger.info("Конвертация в ONNX пропущена (отключена в конфиге)")

        # Конвертация в TensorRT
        if config.conversion.convert_to_tensorrt:
            logger.info("Начинаем конвертацию в TensorRT...")

            # Для TensorRT нужна ONNX модель
            onnx_path = config.onnx.output_path
            if not Path(onnx_path).exists():
                logger.error(
                    f"ONNX модель не найдена: {onnx_path}. Сначала выполните конвертацию в ONNX."
                )
                return

            trt_success = convert_to_tensorrt(
                onnx_path=onnx_path,
                output_path=config.tensorrt.output_path,
                precision=config.tensorrt.precision,
                max_batch_size=config.tensorrt.max_batch_size,
                max_workspace_size=config.tensorrt.max_workspace_size,
            )

            if trt_success:
                logger.info("Конвертация в TensorRT завершена успешно")
            else:
                logger.error("Ошибка при конвертации в TensorRT")
        else:
            logger.info("Конвертация в TensorRT пропущена (отключена в конфиге)")

        logger.info("Пайплайн конвертации модели завершен")

    except Exception as e:
        logger.error(f"Критическая ошибка в пайплайне: {str(e)}")
        raise


if __name__ == "__main__":
    main()
