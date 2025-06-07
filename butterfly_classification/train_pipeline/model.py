import subprocess
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import mlflow
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


class ButterflyClassifier(pl.LightningModule):
    """PyTorch Lightning модуль для классификации бабочек."""

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        model_name: str = "resnet50",
        pretrained: bool = True,
        scheduler: str = "reduce_on_plateau",
        class_names: List[str] = None,
        plots_dir: str = "plots",
        **kwargs,
    ):
        """
        Args:
            num_classes: Количество классов
            learning_rate: Скорость обучения
            weight_decay: Коэффициент регуляризации L2
            model_name: Название модели (resnet18, resnet50, efficientnet_b0, etc.)
            pretrained: Использовать ли предобученную модель
            scheduler: Тип планировщика скорости обучения
            class_names: Названия классов
            plots_dir: Директория для сохранения графиков
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)

        # Создаем модель
        self.model = self._create_model(model_name, num_classes, pretrained)

        # Функция потерь
        self.criterion = nn.CrossEntropyLoss()

        # Метрики для обучения
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # Метрики для валидации
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # Для отслеживания метрик по эпохам
        self.epoch_metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "learning_rate": [],
        }

    def on_train_start(self):
        """Вызывается в начале обучения для логирования гиперпараметров и версии кода."""
        # Логируем гиперпараметры
        hyperparams = {
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "model_name": self.hparams.model_name,
            "pretrained": self.hparams.pretrained,
            "scheduler": self.scheduler,
        }

        # Получаем git commit id
        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
            hyperparams["git_commit_id"] = git_commit
        except Exception:
            hyperparams["git_commit_id"] = "unknown"

        # Логируем в MLflow
        if self.logger and hasattr(self.logger, "experiment"):
            for key, value in hyperparams.items():
                mlflow.log_param(key, value)

    def _create_model(
        self, model_name: str, num_classes: int, pretrained: bool
    ) -> nn.Module:
        """Создание модели."""
        if model_name.startswith("resnet"):
            if model_name == "resnet18":
                model = models.resnet18(pretrained=pretrained)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == "resnet50":
                model = models.resnet50(pretrained=pretrained)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == "resnet101":
                model = models.resnet101(pretrained=pretrained)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                raise ValueError(f"Неподдерживаемая модель ResNet: {model_name}")

        elif model_name.startswith("efficientnet"):
            if model_name == "efficientnet_b0":
                model = models.efficientnet_b0(pretrained=pretrained)
                model.classifier[1] = nn.Linear(
                    model.classifier[1].in_features, num_classes
                )
            else:
                raise ValueError(f"Неподдерживаемая модель EfficientNet: {model_name}")

        else:
            raise ValueError(f"Неподдерживаемая модель: {model_name}")

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход."""
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> Dict[str, Any]:
        """Шаг обучения."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Вычисляем метрики
        preds = torch.argmax(outputs, dim=1)
        self.train_accuracy(preds, labels)
        self.train_f1(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)

        # Логируем метрики
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Dict[str, Any]:
        """Шаг валидации."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Вычисляем метрики
        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)

        # Логируем метрики
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """Сбор метрик для построения графиков."""
        # Получаем текущие метрики
        train_loss = self.trainer.callback_metrics.get("train/loss_epoch", 0)
        train_acc = self.trainer.callback_metrics.get("train/accuracy", 0)
        train_f1 = self.trainer.callback_metrics.get("train/f1", 0)
        val_loss = self.trainer.callback_metrics.get("val/loss", 0)
        val_acc = self.trainer.callback_metrics.get("val/accuracy", 0)
        val_f1 = self.trainer.callback_metrics.get("val/f1", 0)

        # Получаем текущий learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        # Сохраняем метрики
        self.epoch_metrics["train_loss"].append(float(train_loss))
        self.epoch_metrics["train_accuracy"].append(float(train_acc))
        self.epoch_metrics["train_f1"].append(float(train_f1))
        self.epoch_metrics["val_loss"].append(float(val_loss))
        self.epoch_metrics["val_accuracy"].append(float(val_acc))
        self.epoch_metrics["val_f1"].append(float(val_f1))
        self.epoch_metrics["learning_rate"].append(current_lr)

        # Логируем learning rate
        self.log("train/learning_rate", current_lr, on_epoch=True)

    def on_train_end(self):
        """Создание и сохранение графиков в конце обучения."""
        self._create_training_plots()

    def _create_training_plots(self):
        """Создание графиков обучения."""
        epochs = range(1, len(self.epoch_metrics["train_loss"]) + 1)

        # График 1: Loss curves
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(
            epochs,
            self.epoch_metrics["train_loss"],
            "b-",
            label="Train Loss",
            linewidth=2,
        )
        plt.plot(
            epochs, self.epoch_metrics["val_loss"], "r-", label="Val Loss", linewidth=2
        )
        plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График 2: Accuracy curves
        plt.subplot(2, 2, 2)
        plt.plot(
            epochs,
            self.epoch_metrics["train_accuracy"],
            "b-",
            label="Train Accuracy",
            linewidth=2,
        )
        plt.plot(
            epochs,
            self.epoch_metrics["val_accuracy"],
            "r-",
            label="Val Accuracy",
            linewidth=2,
        )
        plt.title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График 3: F1 Score curves
        plt.subplot(2, 2, 3)
        plt.plot(
            epochs, self.epoch_metrics["train_f1"], "b-", label="Train F1", linewidth=2
        )
        plt.plot(
            epochs, self.epoch_metrics["val_f1"], "r-", label="Val F1", linewidth=2
        )
        plt.title("Training and Validation F1 Score", fontsize=14, fontweight="bold")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График 4: Learning Rate
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.epoch_metrics["learning_rate"], "g-", linewidth=2)
        plt.title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Сохраняем график
        plot_path = self.plots_dir / "training_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Логируем график в MLflow
        if self.logger and hasattr(self.logger, "experiment"):
            mlflow.log_artifact(str(plot_path))

    def configure_optimizers(self) -> Dict[str, Any]:
        """Настройка оптимизатора и планировщика."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.scheduler == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "frequency": 1,
                },
            }
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=50)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

    def predict_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Шаг предсказания."""
        images, _ = batch
        outputs = self(images)
        return torch.softmax(outputs, dim=1)
