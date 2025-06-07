import logging
from pathlib import Path
from typing import Dict, List, Optional

import onnx
import torch

try:
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logging.warning(
        "TensorRT not available. Install tensorrt package for TensorRT conversion."
    )

logger = logging.getLogger(__name__)


def convert_to_onnx(
    model: torch.nn.Module,
    input_shape: List[int],
    output_path: str,
    opset_version: int = 11,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> bool:
    """
    Конвертирует PyTorch модель в ONNX формат.

    Args:
        model: PyTorch модель для конвертации
        input_shape: Форма входных данных [batch_size, channels, height, width]
        output_path: Путь для сохранения ONNX модели
        opset_version: Версия ONNX opset
        dynamic_axes: Динамические оси для переменного размера батча

    Returns:
        bool: True если конвертация успешна, False иначе
    """
    try:
        # Создаем директорию если не существует
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Переводим модель в режим оценки
        model.eval()

        # Отключаем градиенты
        torch.no_grad().__enter__()

        # Создаем dummy input
        dummy_input = torch.randn(*input_shape)

        # Проверяем работу модели перед экспортом
        logger.info("Проверка модели перед экспортом...")
        with torch.no_grad():
            test_output = model(dummy_input)
            logger.info(f"Размер выходного тензора: {test_output.shape}")

        logger.info("Начинаем экспорт в ONNX...")

        # Экспортируем в ONNX с дополнительными настройками
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,  # Экспортировать параметры модели
            opset_version=opset_version,
            do_constant_folding=True,  # Константное свертывание для оптимизации
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,  # Явно указываем режим eval
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )

        logger.info("ONNX экспорт завершен, проверяем модель...")

        # Проверяем корректность ONNX модели
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        logger.info(f"Модель успешно конвертирована в ONNX: {output_path}")

        # Выводим информацию о модели
        logger.info(
            f"Размер ONNX файла: {Path(output_path).stat().st_size / (1024*1024):.2f} MB"
        )

        return True

    except Exception as e:
        logger.error(f"Ошибка при конвертации в ONNX: {str(e)}")
        import traceback

        logger.error(f"Детали ошибки: {traceback.format_exc()}")
        return False


def convert_to_tensorrt(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
    max_workspace_size: int = 1073741824,
) -> bool:
    """
    Конвертирует ONNX модель в TensorRT формат.

    Args:
        onnx_path: Путь к ONNX модели
        output_path: Путь для сохранения TensorRT engine
        precision: Точность вычислений ("fp32", "fp16", "int8")
        max_batch_size: Максимальный размер батча
        max_workspace_size: Максимальный размер рабочей области в байтах

    Returns:
        bool: True если конвертация успешна, False иначе
    """
    if not TRT_AVAILABLE:
        logger.error("TensorRT недоступен. Установите tensorrt пакет.")
        return False

    try:
        # Создаем директорию если не существует
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Создаем TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Создаем builder
        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()

        # Настраиваем precision
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)

        # Настраиваем workspace size
        config.max_workspace_size = max_workspace_size

        # Создаем network
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )

        # Парсим ONNX модель
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                logger.error("Ошибка при парсинге ONNX модели")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return False

        # Настраиваем оптимизационный профиль для динамического батча
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_shape = input_tensor.shape

        # Устанавливаем минимальный, оптимальный и максимальный размеры
        min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
        opt_shape = (
            max_batch_size // 2,
            input_shape[1],
            input_shape[2],
            input_shape[3],
        )
        max_shape = (max_batch_size, input_shape[1], input_shape[2], input_shape[3])

        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Строим engine
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.error("Не удалось построить TensorRT engine")
            return False

        # Сохраняем engine
        with open(output_path, "wb") as f:
            f.write(serialized_engine)

        logger.info(f"Модель успешно конвертирована в TensorRT: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Ошибка при конвертации в TensorRT: {str(e)}")
        return False
