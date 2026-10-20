# Распознавание нестандартных ГРЗ РФ (тип 1, 1А — квадратные, 1Б — жёлтые)

Полуфинал «Искусственный интеллект и анализ данных» (куратор — АИС Город). Полностью офлайновое решение:
каталог изображений → CSV `image;plate_num;plate_type;confidence`. Только CPU, зависимости инференса: **numpy, opencv-python**.

> **Состояние.** Модели обучены и измерены **только на синтетике**. Реальная часть датасета пуста (её нужно снять и разметить
> вручную, см. ниже) — качество на реальных кадрах не измерялось. Подробности и числа — `docs/explanatory_note.docx`.

## Запуск (точная команда)
```bash
pip install -r requirements.txt
python run.py --images /путь/к/каталогу/с/изображениями --out result.csv
```
Путь можно задать и переменными окружения `PLATES_IMAGES`, `PLATES_OUT`. Порог уверенности берётся из `models/verifier_threshold.txt`
(переопределяется `--threshold`). Для знаков, которых нет на кадре, строка не добавляется. Все веса — в `models/` (≈1 МБ),
обращений в сеть нет. Модуль `anpr/gost.py` подключает `dataset/generator/gostplate.py` (геометрия слотов знака), поэтому
каталог `dataset/generator/` должен лежать рядом с `anpr/`.

### В Google Colab
```python
!unzip -q plates_solution.zip && cd plates_solution && pip install -q -r requirements.txt
!cd plates_solution && python run.py --images /content/test_images --out /content/result.csv
```

## Состав
```
run.py                      точка входа
anpr/                       detect.py (кандидаты), recognize.py (выпрямление, Витерби по маске), pipeline.py
models/                     recognizer.npz, verifier.npz, verifier_threshold.txt
train/                      train_recognizer.py, train_verifier.py, synth_crops.py
tools/                      evaluate.py, diagnose.py, eval_detect.py, eval_recognizer.py,
                            validate_dataset.py, annotate.py (ручная разметка своих фото), blur_faces.py
dataset/                    датасет по структуре ТЗ (README = датащит, LICENSE, generator/, meta.csv, labels/, images/)
docs/                       explanatory_note.docx / .pdf (до 5 страниц), make_report.py
```

## Воспроизведение
```bash
pip install -r requirements-train.txt
python dataset/generator/generate_synthetic.py --out dataset --n 5000 --seed 20261014 --meta dataset/meta.csv
python train/train_recognizer.py --n 16000 --seed 3 --epochs 20 --hidden 320 160     # ~12 мин на 2 vCPU
python train/train_verifier.py --n 1500 --seed 31                                     # ~10 мин, пишет порог
python tools/validate_dataset.py --root dataset --report dataset/validation_report.md
```
Оценка на отложенных кадрах: `python run.py --images EVAL_DIR --out pred.csv` и
`python tools/evaluate.py --pred pred.csv --gt dataset/meta.csv --images-dir EVAL_DIR`
(на кадрах `syn_04000…syn_04999`: точность 0.93, полнота 0.32 в целом / 0.62 для знаков ≥120 px, 75 мс/изобр. на 2 vCPU).

## Внешние данные (атрибуция CC BY 4.0)

Распознаватель символов дообучен на реальных вырезках российских номеров из набора
[Russian License Plate Characters — 23 Classes](https://www.kaggle.com/datasets/dimahkiin/russian-license-plate-characters-23-classes)
(автор dimahkiin, лицензия CC BY 4.0). Использован только раздел `val` (5000 вырезок, из них 3123 с номером по маске ГРЗ);
20% номеров отложены для проверки по тексту номера. Разметка символов в этом наборе сделана автором с помощью модели —
это внешние данные, они НЕ входят в `dataset/` и не выдаются за собственную ручную разметку.
Скрипты: `train/real_crops.py`, `train/train_recognizer_real.py` (обучение), `tools/eval_real_crops.py` (проверка).
Старые веса только на синтетике — `models/recognizer_synth_only.npz`.

## Что нужно сделать до сдачи
1. **Реальные данные.** Снять собственные фото (парковки, дворы, видеорегистратор) и разметить:
   `python tools/annotate.py --src мои_фото --root dataset --source own_photo --license own_photo --conditions day`
   (клики по 4 углам знака, ручной ввод номера, `b` — размыть лица). Нужно ≥150 изобр. type1a (≥50 номеров), ≥300 type1b (≥100),
   ≥50 negative. Номера вводит человек: автоматическая разметка чужой моделью запрещена ТЗ.
2. **Дообучение на реальных кадрах** (замена/добавление данных в `train/`) и повторная оценка — реальное качество неизвестно.
3. **Валидатор организаторов** (`validate_dataset.py` из ТЗ) прогнать на датасете и приложить отчёт; формат txt-меток в ТЗ не описан.
4. В `dataset/LICENSE` заменить уведомление дословным текстом CC BY 4.0.
5. Замерить время на референсной конфигурации (i5-7600, GTX 1050 Ti): среднее время печатается в конце прогона `run.py`.
