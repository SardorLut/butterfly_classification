"""
Универсальный загрузчик моделей для различных форматов.
"""

import logging
import os
from pathlib import Path
from typing import List

import numpy as np

# PyTorch Lightning
import torch

from butterfly_classification.train_pipeline.model import ButterflyClassifier

# ONNX
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# TensorRT
try:
    import pycuda.driver as cuda
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Универсальная обертка для моделей разных форматов."""

    def __init__(self, model_path: str, class_names: List[str] = None):
        self.model_path = model_path
        self.class_names = class_names
        self.model_type = self._detect_model_type()
        self.model = None
        self.device = "cpu"

        # Загружаем модель
        self._load_model()

    def _detect_model_type(self) -> str:
        """Определяет тип модели по расширению файла."""
        extension = Path(self.model_path).suffix.lower()

        if extension == ".ckpt":
            return "pytorch_lightning"
        elif extension == ".onnx":
            return "onnx"
        elif extension in [".engine", ".trt"]:
            return "tensorrt"
        elif extension in [".pth", ".pt"]:
            return "pytorch"
        else:
            raise ValueError(f"Неподдерживаемый формат модели: {extension}")

    def _load_model(self):
        """Загружает модель в зависимости от типа."""
        if self.model_type == "pytorch_lightning":
            self._load_pytorch_lightning()
        elif self.model_type == "onnx":
            self._load_onnx()
        elif self.model_type == "tensorrt":
            self._load_tensorrt()
        elif self.model_type == "pytorch":
            self._load_pytorch()
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")

    def _load_pytorch_lightning(self):
        """Загружает PyTorch Lightning модель."""
        logger.info(f"Загружаем PyTorch Lightning модель: {self.model_path}")
        self.model = ButterflyClassifier.load_from_checkpoint(self.model_path)
        self.model.eval()

        # Получаем названия классов
        if not self.class_names:
            self.class_names = self.model.class_names

        # Определяем устройство
        self.device = next(self.model.parameters()).device
        logger.info(f"Модель загружена на устройство: {self.device}")

    def _load_onnx(self):
        """Загружает ONNX модель."""
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX Runtime не установлен. Установите: pip install onnxruntime"
            )

        logger.info(f"Загружаем ONNX модель: {self.model_path}")

        # Определяем провайдеры (GPU или CPU)
        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers.insert(0, "CUDAExecutionProvider")

        self.model = ort.InferenceSession(self.model_path, providers=providers)
        self.device = (
            "cuda" if "CUDAExecutionProvider" in self.model.get_providers() else "cpu"
        )

        logger.info(f"ONNX модель загружена, провайдеры: {self.model.get_providers()}")
        logger.info(f"Устройство: {self.device}")

    def _load_tensorrt(self):
        """Загружает TensorRT модель."""
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT не установлен или недоступен")

        logger.info(f"Загружаем TensorRT модель: {self.model_path}")

        # Создаем TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Загружаем engine
        with open(self.model_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.model = runtime.deserialize_cuda_engine(f.read())

        # Создаем контекст выполнения
        self.context = self.model.create_execution_context()

        # Получаем информацию о входных и выходных тензорах
        self.input_binding = self.model.get_binding_name(0)
        self.output_binding = self.model.get_binding_name(1)

        self.device = "cuda"
        logger.info("TensorRT модель загружена на GPU")

    def _load_pytorch(self):
        """Загружает обычную PyTorch модель."""
        logger.info(f"Загружаем PyTorch модель: {self.model_path}")
        # Здесь должна быть логика загрузки обычной PyTorch модели
        # Пока не реализовано
        raise NotImplementedError(
            "Загрузка обычных PyTorch моделей пока не реализована"
        )

    def predict(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Выполняет предсказание.

        Args:
            input_tensor: Входной тензор [batch_size, channels, height, width]

        Returns:
            np.ndarray: Выходные логиты [batch_size, num_classes]
        """
        if self.model_type == "pytorch_lightning":
            return self._predict_pytorch_lightning(input_tensor)
        elif self.model_type == "onnx":
            return self._predict_onnx(input_tensor)
        elif self.model_type == "tensorrt":
            return self._predict_tensorrt(input_tensor)
        else:
            raise ValueError(f"Предсказание для типа {self.model_type} не реализовано")

    def _predict_pytorch_lightning(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Предсказание для PyTorch Lightning модели."""
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            return outputs.cpu().numpy()

    def _predict_onnx(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Предсказание для ONNX модели."""
        # Конвертируем в numpy
        input_array = input_tensor.cpu().numpy()

        # Выполняем инференс
        outputs = self.model.run(None, {"input": input_array})
        return outputs[0]

    def _predict_tensorrt(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Предсказание для TensorRT модели."""
        # Получаем размеры
        input_shape = input_tensor.shape
        output_shape = (
            input_shape[0],
            len(self.class_names) if self.class_names else 1000,
        )

        # Выделяем память на GPU
        d_input = cuda.mem_alloc(input_tensor.nbytes)
        d_output = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)

        # Копируем данные на GPU
        cuda.memcpy_htod(d_input, input_tensor.cpu().numpy().ascontiguousarray())

        # Выполняем инференс
        self.context.execute_v2([int(d_input), int(d_output)])

        # Копируем результат обратно
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)

        return output

    def get_model_info(self) -> dict:
        """Возвращает информацию о модели."""
        info = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": self.device,
            "num_classes": len(self.class_names) if self.class_names else None,
            "class_names": self.class_names,
        }

        if self.model_type == "onnx":
            info["providers"] = self.model.get_providers()
        elif self.model_type == "tensorrt":
            info["max_batch_size"] = self.model.max_batch_size

        return info


def load_model(model_path: str, class_names: List[str] = None) -> ModelWrapper:
    """
    Универсальная функция загрузки модели.

    Args:
        model_path: Путь к файлу модели
        class_names: Список названий классов (опционально)

    Returns:
        ModelWrapper: Обертка модели
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    return ModelWrapper(model_path, class_names)
