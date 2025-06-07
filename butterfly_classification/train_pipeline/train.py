#!/usr/bin/env python3
"""
Скрипт для обучения классификатора бабочек с использованием PyTorch Lightning
"""

import os
import warnings
from pathlib import Path

import hydra
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import MLFlowLogger

from butterfly_classification.train_pipeline.data_module import ButterflyDataModule
from butterfly_classification.train_pipeline.model import ButterflyClassifier

# Игнорируем предупреждения
warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="../../configs", config_name="train_config")
def train(cfg: DictConfig) -> None:
    """Основная функция обучения."""

    print("=" * 50)
    print("BUTTERFLY CLASSIFICATION TRAINING")
    print("=" * 50)
    print(f"Конфигурация:\n{OmegaConf.to_yaml(cfg)}")

    # Устанавливаем seed для воспроизводимости
    pl.seed_everything(cfg.seed, workers=True)

    # Создаем директории для сохранения результатов
    Path(cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.plots_dir).mkdir(parents=True, exist_ok=True)

    # Настраиваем MLflow
    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)

    # Инициализируем DataModule
    print("Инициализация DataModule...")
    data_module = ButterflyDataModule(
        train_csv=cfg.data.train_csv,
        train_images_dir=cfg.data.train_images_dir,
        test_csv=cfg.data.test_csv,
        test_images_dir=cfg.data.test_images_dir,
        **cfg.datamodule,
    )

    # Подготавливаем данные для получения информации о классах
    data_module.setup("fit")
    num_classes = data_module.num_classes
    class_names = data_module.get_class_names()

    print(f"Количество классов: {num_classes}")
    print(f"Размер тренировочного набора: {len(data_module.train_dataset)}")
    print(f"Размер валидационного набора: {len(data_module.val_dataset)}")

    # Инициализируем модель
    print(f"Инициализация модели {cfg.model.model_name}...")
    model = ButterflyClassifier(
        num_classes=num_classes,
        class_names=class_names,
        plots_dir=cfg.logging.plots_dir,
        **cfg.model,
    )

    # Настраиваем колбэки
    callbacks = []

    # Early Stopping
    early_stopping = EarlyStopping(**cfg.early_stopping)
    callbacks.append(early_stopping)

    # Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.logging.save_dir, **cfg.checkpoint
    )
    callbacks.append(checkpoint_callback)

    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Rich Progress Bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)

    # Настраиваем логгер
    logger = MLFlowLogger(
        experiment_name=cfg.logging.project_name,
        run_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_tracking_uri,
    )

    # Инициализируем Trainer
    print("Инициализация PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        callbacks=callbacks, logger=logger, deterministic=True, **cfg.trainer
    )

    # Запускаем обучение
    print("Начинаем обучение...")
    trainer.fit(model, data_module)

    # Сохраняем финальную модель
    final_model_path = os.path.join(cfg.logging.save_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)

    print("Обучение завершено!")
    print(f"Лучшая модель сохранена: {checkpoint_callback.best_model_path}")
    print(f"Финальная модель сохранена: {final_model_path}")
    print(f"Логи MLflow: {cfg.logging.mlflow_tracking_uri}")
    print(f"Графики сохранены в: {cfg.logging.plots_dir}/")

    print("\nДля просмотра логов откройте:")
    print(f"{cfg.logging.mlflow_tracking_uri}")

    return trainer, model, data_module


if __name__ == "__main__":
    train()
