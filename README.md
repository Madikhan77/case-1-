# Vehicle Damage Classification (8 Classes)

Этот проект реализует классификацию повреждений автомобилей по изображениям.  
Модель обучается на данных с разметкой в формате VIA (VGG Image Annotator) и поддерживает **8 классов**:

1. `dent` – вмятины / повреждения кузова  
2. `scratch` – царапины, краска  
3. `broken glass` – разбитые стёкла  
4. `lost parts` – потерянные детали  
5. `punctured` – пробоины  
6. `torn` – разрывы  
7. `broken lights` – повреждённые фары/фонари  
8. `non-damage` – без повреждений (изображения без регионов)  

---

## 📂 Структура проекта


dataset/
├── image/
│ └── image/ # train изображения
├── validation/
│ └── validation/ # validation изображения
0Train_via_annos.json # аннотации для train
0Val_via_annos.json # аннотации для validation
train_history.json # история обучения (сохраняется после запуска)
vehide_cls8_resnet50_best.pt # лучший чекпоинт модели
vehide_cls8_resnet50_last.pt # последний чекпоинт модели




---

## 🚀 Основные файлы

- `main()` — точка входа. Запускает обучение (warmup + fine-tune), сохраняет историю, отчёты и чекпоинты.
- **Парсинг разметки**  
  - `extract_raw_label_from_region` — извлекает сырую метку из region.  
  - `raw_to_final` — маппинг сырой метки (rach, tray_son, vo_kinh и др.) в один из 8 финальных классов.  
  - `aggregate_regions_to_final_label` — агрегирует регионы в одно финальное значение по изображению (weighted majority vote).
- **Датасеты и DataLoader**  
  - `VehicleDamageDataset` — класс PyTorch Dataset.  
  - `get_transforms` — аугментации для train/val.  
- **Модель**  
  - `VehicleDamageModel` — ResNet50 (или EfficientNet).  
  - Методы `freeze_backbone` и `unfreeze_backbone` — для фаз warmup/finetune.  
- **Тренировка**  
  - `train_epoch` и `validate_epoch` — обучение и валидация за эпоху.  
  - Используется `CrossEntropyLoss` с весами классов.  
  - `AdamW` как оптимизатор.  
- **Аналитика**  
  - Логируются accuracy, F1, precision, recall, balanced accuracy.  
  - `plot_confusion` сохраняет confusion matrix.  
  - В конце печатается `classification_report` по всем классам.

---

## ⚙️ Гиперпараметры

- `IMG_SIZE = 448` — размер изображения.  
- `BATCH_SIZE = 8`  
- `EPOCHS = 7` (3 warmup + 4 finetune)  
- `BASE_LR = 3e-4` — learning rate для warmup.  
- `FINETUNE_LR = 1e-4` — learning rate для fine-tune.  
- `WEIGHT_DECAY = 1e-4`  
- `MODEL_NAME = "resnet50"`  

---

## 🔧 Как запустить

1. Установить зависимости:
   ```bash
   pip install torch torchvision scikit-learn matplotlib seaborn tqdm pillow
