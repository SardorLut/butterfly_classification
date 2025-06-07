import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ButterflyDataset(Dataset):
    """Dataset для классификации бабочек."""

    def __init__(
        self,
        csv_file: str,
        images_dir: str,
        transform: transforms.Compose = None,
        label_to_idx: Dict[str, int] = None,
        is_test_dataset: bool = False,
    ):
        """
        Args:
            csv_file: Путь к CSV файлу с метками
            images_dir: Путь к папке с изображениями
            transform: Трансформации для изображений
            label_to_idx: Словарь для маппинга меток в индексы
            is_test_dataset: Флаг для тестового датасета без меток
        """
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.is_test_dataset = is_test_dataset

        # Проверяем наличие колонки 'label'
        self.has_labels = "label" in self.data.columns

        # Создаем маппинг классов в индексы, если не предоставлен
        if label_to_idx is None and self.has_labels:
            unique_labels = sorted(self.data["label"].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx if label_to_idx is not None else {}

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        """Получение элемента по индексу."""
        row = self.data.iloc[idx]
        image_path = os.path.join(self.images_dir, row["filename"])

        # Загружаем изображение
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Ошибка загрузки изображения {image_path}: {e}")
            # Создаем черное изображение в качестве fallback
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # Применяем трансформации
        if self.transform:
            image = self.transform(image)

        # Получаем индекс класса или возвращаем dummy значение для тестового датасета
        if self.has_labels and not self.is_test_dataset:
            label = self.label_to_idx[row["label"]]
        else:
            # Для тестового датасета без меток возвращаем 0 (dummy label)
            label = 0

        return image, label

    def get_class_names(self) -> List[str]:
        """Возвращает список названий классов."""
        return [self.idx_to_label[i] for i in range(self.num_classes)]


def get_transforms(train: bool = True) -> transforms.Compose:
    """Получение трансформаций для обучения и валидации."""

    if train:
        # Трансформации для обучения с аугментацией
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        # Трансформации для валидации/тестирования
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
