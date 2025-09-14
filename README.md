# 🚗 Vehicle Damage Classification (8 Classes)

Модель классифицирует тип повреждения автомобиля по фото.  
Разметка — **VIA (VGG Image Annotator)**. Поддерживаемые классы:

1. `dent` — вмятины / деформация кузова  
2. `scratch` — царапины / покраска  
3. `broken glass` — треснувшие/разбитые стёкла  
4. `lost parts` — отсутствующие детали  
5. `punctured` — проколы/пробоины  
6. `torn` — разрывы  
7. `broken lights` — повреждённые фары/фонари  
8. `non-damage` — без повреждений *(если `regions` пустой)*  


## 📂 Структура проекта

```
dataset/
├─ image/
│ └─ image/ # train изображения
├─ validation/
│ └─ validation/ # validation изображения

0Train_via_annos.json # аннотации (train, VIA)
0Val_via_annos.json # аннотации (val, VIA)
train_history.json # история обучения
vehide_cls8_resnet50_best.pt # лучший чекпоинт модели (скачать с Google Drive)
vehide_cls8_resnet50_last.pt # последний чекпоинт модели
main.py # веб-демо на Streamlit
index.html # слайды с результатами
in.py / main.py # обучение модели
```





## ☁️ Весa модели

Обученный чекпоинт **`vehide_cls8_resnet50_best.pt`** хранится на Google Drive.  
Скачайте его и положите в корень проекта.  

🔗 Ссылка: **[Google Drive (см. внутреннюю ссылку команды)](https://drive.google.com/)**  

---

## ⚙️ Установка окружения

Рекомендуется Python 3.10–3.11.  
Минимальные зависимости:

```
pip install torch torchvision numpy pillow scikit-learn matplotlib seaborn tqdm streamlit
Или через файл:

bash
Копировать код
pip install -r requirements.txt
🚀 Быстрый старт (Streamlit демо)
Запустить веб-приложение:

bash
Копировать код
streamlit run main.py
Браузер откроется по адресу: http://localhost:8501

Загрузите фото → получите класс и вероятности по всем 8 классам

🖼️ Презентация (слайды)
Откройте файл index.html двойным кликом.
Навигация: стрелки на клавиатуре или кнопки «Назад / Вперёд».

Ключевые метрики (после 7 эпох):

Train Accuracy: 70.52%

Validation Accuracy: 67.60%

Macro F1: 0.417

Macro Precision: 0.390

Macro Recall: 0.471

Balanced Accuracy: 0.754

⚡ Модель обучалась всего 7 эпох (3 warmup + 4 finetune). Большее число эпох + больше данных обычно повышают метрики.


Про датасет

Формат: VIA (VGG Image Annotator) — для каждой картинки есть список regions с метками.

Язык аннотаций: вьетнамский (rach, mop_lom, vo_kinh), маппятся в 8 финальных классов:

rach, paint, paint_damage, tray_son → scratch

mop_lom, door_dent, bumper_dent, body_damage, thung → dent

vo_kinh, broken_glass, glass_shatter → broken glass

mat_bo_phan → lost parts

broken_lights, headlight_damage, tail_lamp_broken, be_den → broken lights

punctured → punctured

torn → torn

regions == [] → non-damage

⚠️ Дисбаланс: много scratch и dent, мало punctured, torn, non-damage. Это снижает Macro F1 → рекомендуется oversampling и data augmentation.

✅ Выводы

Даже за 7 эпох модель достигла 67.6% на валидации.

Balanced Accuracy = 0.754 показывает, что сеть учитывает редкие классы.

Для улучшения: больше данных, балансировка классов, больше эпох, новые аугментации.

📄 Лицензия

Разметка VIA создаётся вручную аннотаторами.

Судя по тегам (rach, mop_lom, vo_kinh), датасет собран во Вьетнаме (страховые/исследовательские проекты).

Использовать данные следует в соответствии с условиями источника.
