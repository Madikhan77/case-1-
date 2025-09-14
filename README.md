Vehicle Damage Classification (8 Classes)

Модель классифицирует тип повреждения автомобиля по фото.
Разметка — VIA (VGG Image Annotator). Поддерживаемые классы:

dent — вмятины / деформация кузова

scratch — царапины / покраска

broken glass — треснувшие/разбитые стёкла

lost parts — отсутствующие детали

punctured — проколы/пробоины

torn — разрывы

broken lights — повреждённые фары/фонари

non-damage — без повреждений (когда regions пустой)

📦 Что внутри репозитория
dataset/
├─ image/
│  └─ image/             # train изображения
├─ validation/
│  └─ validation/        # validation изображения
0Train_via_annos.json    # аннотации (train, формат VIA)
0Val_via_annos.json      # аннотации (val, формат VIA)
train_history.json       # история обучения (создаётся после запуска)
vehide_cls8_resnet50_best.pt  # лучший чекпоинт (скачать, см. ниже)
vehide_cls8_resnet50_last.pt  # последний чекпоинт (опционально)
app_streamlit.py         # веб-демо на Streamlit
presentation.html        # слайды с результатами
train.py / main.py       # обучение модели


Если каких-то файлов нет — это нормально: они появятся после обучения или их нужно скачать (см. следующий раздел).

☁️ Где взять весa модели

Обученный чекпоинт vehide_cls8_resnet50_best.pt лежит у тебя на Google Drive.
Скачай его и положи рядом с кодом (в корень проекта).

Ссылка на Google Drive: (вставь сюда свою ссылку на файл)
(Мы не публикуем ссылку здесь, но команда знает её. Без этого файла Streamlit-демо не запустится.)

Проверка структуры после скачивания:

.
├─ app_streamlit.py
├─ presentation.html
├─ vehide_cls8_resnet50_best.pt  # <-- должен существовать
└─ ...

⚙️ Установка и окружение

Python 3.10–3.11 рекомендуется.

Минимальные зависимости:

pip install -U pip
pip install torch torchvision numpy pillow scikit-learn matplotlib seaborn tqdm streamlit


(или положи наш короткий requirements.txt и выполни pip install -r requirements.txt)

🚀 Быстрый старт (Streamlit демо)

Интерактивное тестирование одной картинки через веб-интерфейс.

streamlit run app_streamlit.py


Откроется браузер: http://localhost:8501

Загрузите фото → получите класс + вероятности по всем 8 классам.

Если у тебя OneDrive/путь с пробелами/символами типа & — это нормально; команду streamlit run запускай из корня проекта.

🖼️ Презентация (слайды)

Готовый HTML-файл с результатами и описанием датасета:

Открой presentation.html двойным кликом.

Переключение слайдов — кнопками «Назад/Вперёд» или стрелками ←/→.

Ключевые метрики (после 7 эпох):

Train Acc: 70.52%

Val Acc: 67.60%

Macro F1: 0.417

Macro Precision: 0.390

Macro Recall: 0.471

Balanced Acc: 0.754

Модель обучалась всего 7 эпох (3 warmup + 4 finetune). Большее число эпох, дополнительные данные и балансировка классов обычно улучшают метрики.

🧠 Обучение (если хочешь переобучить)

Параметры по умолчанию:

IMG_SIZE=448, BATCH_SIZE=8

EPOCHS=7 (3 warmup + 4 finetune)

BASE_LR=3e-4, FINETUNE_LR=1e-4

Оптимизатор: AdamW

Лосс: CrossEntropyLoss с весами классов (по обратной частоте)

Запуск (пример):

python train.py


После обучения появятся/обновятся:

vehide_cls8_resnet50_best.pt

vehide_cls8_resnet50_last.pt

train_history.json

confusion_matrix.png

подробный classification_report в консоли

🧾 Про датасет

Формат аннотаций: VIA (VGG Image Annotator) — список изображений с массивом regions, у каждого региона есть class/label и координаты.

Особенность: в аннотациях встречаются вьетнамские теги (например, rach, mop_lom, vo_kinh), которые мы маппим на финальные 8 классов:

rach, paint, paint_damage, tray_son → scratch

mop_lom, body_damage, door_dent, bumper_dent, thung → dent

vo_kinh, broken_glass, glass_shatter → broken glass

mat_bo_phan → lost parts

broken_lights, headlight_damage, tail_lamp_broken, be_den → broken lights

punctured → punctured

torn → torn

regions == [] → non-damage

В текущих JSON часто нет пустых regions, поэтому non-damage может отсутствовать в обучении. Добавь чистые снимки без разметки (или собери кадры с regions: []), чтобы модель научилась классу non-damage.

Дисбаланс: много scratch/dent, мало broken lights, практически нет punctured/torn/non-damage. Это снижает Macro F1; используем веса классов, и при необходимости — WeightedRandomSampler/oversampling.

🔍 Инференс (скриптом)

Мини-пример (одна картинка, как в test_one_image.py):

import torch
from PIL import Image
from torchvision import transforms

from model import VehicleDamageModel, FINAL_CLASSES  # твой модуль

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 448
TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = VehicleDamageModel(num_classes=len(FINAL_CLASSES), model_name="resnet50").to(DEVICE)
ckpt = torch.load("vehide_cls8_resnet50_best.pt", map_location=DEVICE)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()

img = Image.open("test.jpg").convert("RGB")
x = TF(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    out = model(x)
    probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    pred = FINAL_CLASSES[int(probs.argmax())]

print("Prediction:", pred)

🛠️ Трудности и как их обойти

Дисбаланс классов → веса классов, oversampling, таргетированные аугментации для редких классов.

Отсутствие non-damage → добавить снимки без разметки (regions: []).

Похожие классы путаются (dent vs punctured) → больше примеров, более чёткие правила разметки.

Мало эпох (7) → увеличить EPOCHS, добавить early-stopping/ReduceLROnPlateau.

🧩 Частые проблемы

PowerShell: “Амперсанд (&) не разрешен…”
Путь содержит &. Либо возьми путь в кавычки, либо запускай команды из директории проекта:

cd "C:\Users\...\InDrive"
streamlit run app_streamlit.py


Uvicorn не нужен (мы используем Streamlit). Если решишь вернуться к API — лучше держать FastAPI в отдельном файле.

📄 Лицензия и авторство датасета

Разметка VIA обычно создаётся вручную командами аннотаторов.

Судя по тегам (rach, mop_lom, vo_kinh), датасет собирался во Вьетнаме (страховые/исследовательские группы).

Если публикуешь код и веса публично — добавь лицензию и условия использования данных согласно договорённостям/источнику.
