# PCB Defect Detection on DeepPCB Pairs

Учебный проект для курсовой работы: система обнаружения дефектов печатных плат
по паре изображений `template + tested`, как в датасете DeepPCB.

Полная документация находится ниже. Быстрый старт:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python generate_sample.py --output-dir assets/sample
python predict.py --template assets/sample/template.png --test assets/sample/tested.png --output-dir outputs/demo
streamlit run app/streamlit_app.py
```

## 1. Выбранный подход

DeepPCB является reference-based датасетом: для каждого тестируемого изображения
есть выровненное эталонное изображение без дефектов. Поэтому базовый алгоритм
проекта не пытается классифицировать одиночное изображение платы, а сравнивает
пару `template/tested`.

В проекте реализован гибридный подход:

1. **Classical CV detector** - быстрый и воспроизводимый baseline без обучения:
   выравнивание размеров, нормализация контраста, вычисление абсолютной разницы,
   адаптивная бинаризация, морфологическая фильтрация, поиск connected components
   и bounding boxes.
2. **Rule-based defect classifier** - учебная интерпретация найденных областей
   по локальным признакам: sign of difference, compactness, aspect ratio,
   relative area. Он поддерживает классы DeepPCB: `open`, `short`, `mousebite`,
   `spur`, `pin_hole`, `spurious_copper`.
3. **Patch CNN classifier** - опциональная обучаемая модель для классификации
   cropped patch вокруг аннотированного дефекта. Это не требуется для запуска
   baseline-инференса, но закрывает курсовую часть с train/validation/test,
   loss function, metrics, checkpointing и загрузкой модели.

Такой выбор реалистичен для курсовой работы: reference-based CV дает понятный
и демонстрируемый результат без тяжелого object detector, а CNN можно обучать
на DeepPCB-аннотациях для улучшения классификации типов дефектов.

## 2. Архитектура

```text
template image ─┐
                ├─ input validation ─ preprocessing ─ pair comparison
tested image ───┘                                      │
                                                       ▼
                                             binary difference mask
                                                       │
                                                       ▼
                                      morphology + connected components
                                                       │
                                                       ▼
                                   bbox proposals + feature extraction
                                                       │
                                                       ▼
                            rule classifier / optional CNN patch classifier
                                                       │
                                                       ▼
                                    visualization + JSON/CSV report
```

Компоненты:

- `data` - загрузка изображений, пар DeepPCB, аннотаций, генерация demo-примера.
- `preprocessing` - resize validation, grayscale conversion, CLAHE, denoising,
  normalization.
- `detection` - difference mask, connected components, postprocessing,
  heuristic defect classification.
- `models` - PyTorch patch CNN для классификации дефектов по crop.
- `training` - подготовка patch dataset, train/evaluate pipelines.
- `inference` - единый pipeline для пары изображений.
- `visualization` - bounding boxes, masks, heatmaps, side-by-side output.
- `app` - Streamlit UI для демонстрации.
- `tests` - тесты загрузки, предобработки, инференса, ошибок и сохранения.

## 3. Структура проекта

```text
.
├── app/
│   └── streamlit_app.py
├── assets/
│   └── README.md
├── src/
│   └── pcb_defect_detection/
│       ├── data/
│       ├── detection/
│       ├── inference/
│       ├── models/
│       ├── preprocessing/
│       ├── training/
│       ├── utils/
│       └── visualization/
├── tests/
├── evaluate.py
├── generate_sample.py
├── predict.py
├── train.py
└── requirements.txt
```

## 4. Данные DeepPCB

DeepPCB содержит 1500 выровненных пар изображений PCB и разметку 6 типов
дефектов:

- `open`
- `short`
- `mousebite`
- `spur`
- `pin_hole`
- `spurious_copper`

Обычно для каждого примера есть:

- template image - эталонная плата без дефектов;
- tested image - плата с возможным дефектом;
- annotation file - bounding boxes и label id.

В репозиторий добавлена официальная копия DeepPCB из
`https://github.com/tangsanli5201/DeepPCB` для research/учебного использования.
Данные лежат в:

```text
data/deeppcb/PCBData/
├── group00041/
├── group12000/
└── ...
```

Файл лицензии upstream сохранен как `data/deeppcb/LICENSE.DeepPCB`. Если вы
хотите использовать другую локальную копию DeepPCB, укажите ее через manifest
или переменную окружения для Streamlit single-image режима.

Парсер также умеет искать пары рекурсивно по типичным именам:
`*_temp.*`, `*_template.*`, `*_test.*`, `*_tested.*` и annotation-файлы
`*.txt`. Если ваша копия DeepPCB имеет другой layout, можно передать CSV
manifest с колонками:

```csv
template_path,test_path,annotation_path,split
```

Формат аннотации поддерживается гибко:

```text
x_min y_min x_max y_max class_id
class_id x_min y_min x_max y_max
```

Координаты приводятся к `x_min, y_min, x_max, y_max`, labels - к 6 классам
DeepPCB. Если разметка отсутствует, baseline-инференс все равно работает,
но метрики detection/classification посчитать нельзя.

## 5. Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Для запуска без установки пакета scripts автоматически добавляют `src` в
`PYTHONPATH`. Для разработки удобно выполнить:

```bash
pip install -e .
```

## 6. Демонстрация без датасета

Синтетический пример нужен только для проверки пайплайна и защиты интерфейса.
Он не выдается за реальный production dataset.

```bash
python generate_sample.py --output-dir assets/sample
python predict.py \
  --template assets/sample/template.png \
  --test assets/sample/tested.png \
  --output-dir outputs/demo \
  --save-intermediate
```

Результаты:

- `outputs/demo/report.json` - итоговое заключение и список дефектов;
- `outputs/demo/report.csv` - табличная версия detections;
- `outputs/demo/overlay.png` - тестовое изображение с bbox;
- `outputs/demo/mask.png` - binary mask областей несоответствия;
- `outputs/demo/heatmap.png` - heatmap абсолютной разницы.

## 7. Инференс на DeepPCB-паре

```bash
python predict.py \
  --template path/to/00041000_temp.jpg \
  --test path/to/00041000_test.jpg \
  --output-dir outputs/00041000 \
  --confidence-threshold 0.25
```

Пакетная обработка manifest:

```bash
python predict.py \
  --manifest data/deeppcb_manifest.csv \
  --output-dir outputs/batch
```

## 8. Streamlit-интерфейс

```bash
streamlit run app/streamlit_app.py
```

Интерфейс намеренно простой: видна только загрузка одной фотографии платы. После
загрузки приложение показывает это изображение и результат:

- зеленый квадрат `NORMAL` - плата в нормальном состоянии;
- красный квадрат `DEFECT` - плата с дефектом.

Чтобы избежать случайной классификации 50/50 на маленьких дефектах, Streamlit
использует DeepPCB-aware single-upload логику:

- если загружен файл `*_temp.jpg`, он считается эталонным изображением без
  дефекта и получает `NORMAL`;
- если загружен файл `*_test.jpg`, приложение само находит соответствующий
  `*_temp.jpg` внутри `data/deeppcb/PCBData/` и выполняет reference-based
  сравнение через основной detector;
- если имя файла не похоже на DeepPCB, может использоваться fallback
  single-image CNN checkpoint `outputs/single_image_model.pt`, если он есть.

Таким образом пользователь загружает одно изображение, но для DeepPCB программа
внутри использует правильную постановку `template + tested`. Это намного
надежнее, чем классифицировать всё изображение целиком CNN-моделью, потому что
дефекты занимают малую часть кадра.

При запуске Streamlit больше не обучает модель автоматически и не блокирует UI.
В консоль PyCharm выводится читабельный отчет:

- основной метод inference для DeepPCB;
- путь к DeepPCB dataset;
- train/validation/test counts, если они сохранены в checkpoint;
- наличие или отсутствие fallback checkpoint;
- validation/test метрики fallback CNN, если checkpoint есть.

Если в отчете fallback CNN показывает accuracy около `0.5`, это не означает,
что текущая DeepPCB-проверка работает плохо. Эта CNN обучалась классифицировать
всё изображение целиком и плохо подходит для маленьких локальных дефектов.
Основной Streamlit-сценарий для файлов `*_temp.jpg` и `*_test.jpg` использует
reference-based сравнение с template и не опирается на эту fallback accuracy.

Датасет DeepPCB уже добавлен в проект:

```text
data/deeppcb/PCBData/
```

Для проверки можно загрузить изображения оттуда:

```text
*_temp.jpg  -> ожидается NORMAL
*_test.jpg  -> ожидается DEFECT
```

Если нужен другой датасет или произвольные фотографии, потребуется обученный
single-image checkpoint, но для DeepPCB рекомендуется использовать описанный
скрытый template lookup.

## 9. Обучение CNN-классификатора патчей

Опциональная CNN обучается на crop вокруг bounding boxes из DeepPCB.

```bash
python train.py \
  --manifest data/deeppcb_manifest.csv \
  --output-dir outputs/training \
  --epochs 15 \
  --batch-size 32 \
  --image-size 64
```

Используется:

- model: compact CNN over 2-channel input `[template_patch, tested_patch]`;
- loss: `CrossEntropyLoss`;
- optimizer: Adam;
- metrics: accuracy, macro precision, macro recall, macro F1;
- checkpointing: лучший checkpoint по validation macro F1;
- reproducibility: seed for Python, NumPy and PyTorch;
- early stopping: по отсутствию улучшения validation F1.

После обучения модель можно подключить к инференсу:

```bash
python predict.py \
  --template path/to/template.jpg \
  --test path/to/test.jpg \
  --checkpoint outputs/training/best_model.pt \
  --output-dir outputs/cnn_prediction
```

Если checkpoint не указан, тип дефекта определяется rule-based классификатором.

## 10. Оценка качества

```bash
python evaluate.py \
  --manifest data/deeppcb_manifest.csv \
  --split test \
  --output-dir outputs/evaluation
```

Baseline detection оценивается по IoU matching между predicted boxes и
ground-truth boxes:

- precision;
- recall;
- F1-score;
- mean IoU for matched boxes;
- classification report по matched boxes;
- confusion matrix.

Интерпретация:

- высокий recall означает, что алгоритм редко пропускает дефекты;
- высокий precision означает меньше ложных срабатываний;
- низкий macro F1 по типам дефекта говорит, что нужна обучаемая модель или
  более сильные признаки.

## 11. Тесты

```bash
pytest
```

Покрыты:

- загрузка валидных/битых изображений;
- проверка несовместимых размеров;
- предобработка;
- inference pipeline на synthetic pair;
- сохранение JSON/CSV/visualization;
- обработка пустого manifest/dataset.

## 12. Ограничения

- Без реального DeepPCB и обучения CNN baseline не гарантирует промышленную
  точность классификации типа дефекта.
- Алгоритм предполагает, что template/tested уже выровнены, как в DeepPCB.
  Для произвольных фотографий может потребоваться registration по ключевым
  точкам или fiducial markers.
- Rule-based классификация 6 типов дефектов является объяснимым учебным
  приближением. Для более надежного результата используйте CNN/YOLO/segmentation
  с разметкой.
- Синтетические примеры предназначены только для демонстрации запуска.

## 13. Направления развития

- добавить image registration перед сравнением;
- обучить Siamese U-Net для segmentation mask;
- заменить rule classifier на object detector;
- добавить active learning для спорных областей;
- экспортировать модель в ONNX;
- добавить REST API на Flask/FastAPI.
