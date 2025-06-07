from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .dataset import ButterflyDataset, get_transforms


class ButterflyDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule для данных бабочек."""

    def __init__(
        self,
        train_csv: str,
        train_images_dir: str,
        test_csv: str = None,
        test_images_dir: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        **kwargs
    ):
        """
        Args:
            train_csv: Путь к CSV файлу с тренировочными данными
            train_images_dir: Путь к папке с тренировочными изображениями
            test_csv: Путь к CSV файлу с тестовыми данными
            test_images_dir: Путь к папке с тестовыми изображениями
            batch_size: Размер батча
            num_workers: Количество рабочих процессов для загрузки данных
            val_split: Доля данных для валидации
        """
        super().__init__()
        self.save_hyperparameters()

        self.train_csv = train_csv
        self.train_images_dir = train_images_dir
        self.test_csv = test_csv
        self.test_images_dir = test_images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        # Атрибуты для хранения датасетов
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.label_to_idx = None
        self.num_classes = None

    def setup(self, stage: Optional[str] = None):
        """Настройка датасетов для каждой стадии."""

        if stage == "fit" or stage is None:
            # Создаем полный тренировочный датасет
            full_dataset = ButterflyDataset(
                csv_file=self.train_csv,
                images_dir=self.train_images_dir,
                transform=get_transforms(train=True),
            )

            # Сохраняем маппинг классов и количество классов
            self.label_to_idx = full_dataset.label_to_idx
            self.num_classes = full_dataset.num_classes

            # Разделяем на тренировочный и валидационный наборы
            val_size = int(len(full_dataset) * self.val_split)
            train_size = len(full_dataset) - val_size

            self.train_dataset, val_dataset_temp = random_split(
                full_dataset, [train_size, val_size]
            )

            # Создаем валидационный датасет с отдельными трансформациями
            self.val_dataset = ButterflyDataset(
                csv_file=self.train_csv,
                images_dir=self.train_images_dir,
                transform=get_transforms(train=False),
                label_to_idx=self.label_to_idx,
            )
            # Применяем те же индексы, что и для валидационного разделения
            val_indices = val_dataset_temp.indices
            self.val_dataset.data = self.val_dataset.data.iloc[val_indices].reset_index(
                drop=True
            )

        if stage == "test" or stage is None:
            if self.test_csv and self.test_images_dir:
                self.test_dataset = ButterflyDataset(
                    csv_file=self.test_csv,
                    images_dir=self.test_images_dir,
                    transform=get_transforms(train=False),
                    label_to_idx=self.label_to_idx,
                    is_test_dataset=True,  # Флаг для тестового датасета без меток
                )

    def train_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для тренировки."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для валидации."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для тестирования."""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_class_names(self):
        """Возвращает названия классов."""
        if self.label_to_idx is None:
            # Загружаем временный датасет для получения маппинга классов
            temp_dataset = ButterflyDataset(
                csv_file=self.train_csv, images_dir=self.train_images_dir
            )
            return temp_dataset.get_class_names()

        return [
            label for label, _ in sorted(self.label_to_idx.items(), key=lambda x: x[1])
        ]
