import os
import shutil
from pathlib import Path

import kagglehub


def main():
    print("Скачивание датасета бабочек с Kaggle...")

    # Скачиваем датасет
    path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")
    print(f"Данные скачаны в: {path}")

    # Создаем целевую папку
    target_dir = Path("data/dataset")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Перемещаем данные в нужную папку
    if os.path.exists(path):
        print(f"Перемещение данных в {target_dir}...")

        # Копируем содержимое скачанной папки
        for item in os.listdir(path):
            source = os.path.join(path, item)
            destination = target_dir / item

            if os.path.isdir(source):
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

        print("Данные успешно перемещены в data/dataset/")

        # Показываем структуру данных
        print("\n📊 Структура данных:")
        for root, dirs, files in os.walk(target_dir):
            level = root.replace(str(target_dir), "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:3]:  # Показываем только первые 3 файла
                print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... и еще {len(files) - 3} файлов")
    else:
        print("Ошибка: данные не найдены")


if __name__ == "__main__":
    main()
