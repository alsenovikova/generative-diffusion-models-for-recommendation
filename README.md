# Generative Diffusion Models for Recommendation

**Ranking-Aligned Contrastive Consistency Distillation (RCCD) for one-step diffusion-based sequential recommendation**

[English](#english) · [Русский](#русский)

---

<a id="english"></a>
## English

### About

This repository contains the code for the Master's thesis **"Generative Diffusion Models for Recommendation"** (Alisa Novikova, HSE University, Faculty of Computer Science, programme 01.04.02 "Applied Mathematics and Informatics", 2026; supervisor — Dmitry I. Ignatov).

A paper based on the results of this thesis, **"Ranking-Aligned Contrastive Consistency Distillation for One-Step Diffusion-Based Sequential Recommendation"** (A. Novikova, D. I. Ignatov), has been accepted to **DAMDID/RCDL 2026**, the XXVIII International Conference on Data Analytics and Management in Data Intensive Domains (Nizhny Novgorod, October 20–23, 2026), Session 6 "Recommender Systems and Scientific Knowledge Services". The paper is a condensed version of the thesis, adapted to the conference requirements.

> **Publication:** _to be updated after the conference_ — venue, volume, pages, DOI.

### Motivation

Diffusion models set a new quality standard for sequential recommendation, but they pay for it with iterative inference: tens of denoising steps per request, which is hard to reconcile with the latency budgets of online serving. Simply truncating the reverse trajectory to one step does not work — the DiffuRec teacher loses roughly half of its ranking quality at NFE = 1.

### Goal

Train a **single-step student** that reproduces the ranking quality of a multi-step latent-diffusion recommender (DiffuRec) at the cost of one forward pass.

Research questions:

- **RQ1.** Can consistency distillation produce a one-step student that approaches the quality of its full-trajectory teacher?
- **RQ2.** Does aligning the distillation objective with retrieval geometry (an in-batch contrastive term) improve on classical consistency distillation, and is the gain statistically reliable?
- **RQ3.** How robust is the student across inference budgets, datasets, random seeds, user history lengths and item popularity?

### Method

RCCD combines three components:

1. **Discrete-time consistency distillation** along the reverse trajectory of the frozen DiffuRec teacher (DDIM-style teacher step, EMA target network).
2. **Residual refinement head** — a lightweight MLP on top of the backbone and coarse head inherited from the teacher. Its last layer is zero-initialised, so at step zero the student exactly reproduces the teacher.
3. **Ranking-aligned InfoNCE term** — an in-batch contrastive loss anchored on the true next-item embedding, aligning the training geometry with inner-product ranking over the full catalogue.

The student architecture is identical for all loss variants (CD-only vs RCCD), so the main ablation isolates the effect of the loss function itself.

### Input data

Three public sequential-recommendation benchmarks:

| Dataset | Users | Items | Interactions | Avg. length | Sparsity |
|---|---:|---:|---:|---:|---:|
| Amazon Beauty | 22,363 | 12,101 | 198,502 | 8.53 | 99.93% |
| Amazon Toys | 19,412 | 11,924 | 167,597 | 8.63 | 99.93% |
| MovieLens-1M | 6,040 | 3,416 | 999,611 | 165.50 | 95.16% |

Each dataset is expected at `<data_root>/<dataset>/dataset.pkl` (dataset names: `toys`, `amazon_beauty`, `ml-1m`; default `data_root` is `../datasets/data`). The pickle is a dict with the keys:

- `train` — `{user_id: [item_id, ...]}`, chronologically ordered history;
- `val` — `{user_id: [item_id]}`, the penultimate interaction;
- `test` — `{user_id: [item_id]}`, the last interaction;
- `smap` — item id mapping (its size defines the catalogue size).

`dataset.pkl` files are not stored in the repository because of their size (see `.gitignore`).

### Outputs

- teacher and student checkpoints (`.pt`);
- training logs and per-epoch metric CSVs;
- JSON with results for all seeds and NFE values (`multiseed_results.json`), hyperparameter-selection results;
- `test_predictions_nfe1.npz` with per-user predictions for stratified analysis;
- tables and figures of the thesis, produced by `analyzes.py`.

Metrics: HR@{5,10,20} and NDCG@{5,10,20} against the full catalogue (leave-last-out protocol), plus per-sample latency in ms.

### Repository structure

```
original_diffurec/          DiffuRec teacher
  diffurec.py                 diffusion process, noise schedules, denoiser
  model.py                    Att_Diffuse_model, losses
  step_sample.py              timestep samplers
  trainer.py                  teacher training and evaluation
  utils.py                    data loaders
  main.py                     teacher training entry point
consistency_diffurec/       RCCD student and experiments
  consistency_diffurec.py     ConsistencyStudent (backbone + refinement head)
  distill_trainer.py          distillation loop and evaluation at a given NFE
  evaluation.py               truncated-DDIM teacher, latency measurement
  distill_main.py             entry point: single run / sweep / multi-seed
  hp_selection.py             hyperparameter selection on selection seeds
  multi_seed_runner.py        final runs on the evaluation seeds
  sanity_check.py             hyperparameter transfer check across datasets
  analyzes.py                 tables, statistical tests and figures
```

### Running

```bash
pip install -r requirements.txt
export PYTHONPATH=original_diffurec:consistency_diffurec

# teacher
python original_diffurec/main.py --dataset toys --data_root ../datasets/data

# a single RCCD run (trains the teacher if --teacher_ckpt is not given)
python consistency_diffurec/distill_main.py --dataset toys \
    --teacher_ckpt checkpoints/teacher_toys.pt \
    --contrast_weight 0.5 --contrast_temperature 0.1

# grid over (beta, tau), including the CD-only baseline
python consistency_diffurec/distill_main.py --dataset toys --sweep --include_baseline

# final runs on the five evaluation seeds with the CD-only baseline
python consistency_diffurec/distill_main.py --dataset toys --multiseed --multiseed_baseline
```

Full pipeline of the thesis: `hp_selection.py` (Toys, selection seeds {2017, 2024, 1997}) → `multi_seed_runner.py` with `--best_beta --best_tau --best_lr` (evaluation seeds {1907, 1977, 2015, 23, 88}) → `sanity_check.py` → `analyzes.py`. The experiments were run in Google Colab on an NVIDIA A100; `hp_selection.py`, `multi_seed_runner.py`, `sanity_check.py` and `analyzes.py` use Google Drive paths from the `DRIVE_BASE` / `*_ROOT` constants at the top of each file — change them for a local run.

### Results

HR@10 (%) at NFE = 1, mean over five seeds:

| | Toys | Beauty | ML-1M |
|---|---:|---:|---:|
| DiffuRec, full trajectory | 7.53 | 7.83 | 26.26 |
| DiffuRec, truncated to 1 step | 3.67 | 4.08 | 13.42 |
| **RCCD, 1 step** | **8.06** | **8.15** | **26.52** |

Per-sample latency (ms, batch 1, A100):

| | Toys | Beauty | ML-1M |
|---|---:|---:|---:|
| DiffuRec, full trajectory | 2.244 | 2.238 | 203.638 |
| **RCCD, 1 step** | **0.070** | **0.068** | **1.299** |
| Speed-up | 32.1× | 32.9× | 156.8× |

Key findings:

- Naive truncation keeps only 48–52% of teacher HR@10; the RCCD student recovers 100.9–110.7% of full-trajectory quality across all metrics and datasets.
- Adding the InfoNCE term to consistency distillation gives a statistically significant improvement in all 18 seed-matched paired comparisons after Holm–Bonferroni correction; the gain is largest at large cutoffs and on long-tail items.
- Student quality is nearly flat in NFE, seed variance is lower than that of the truncated teacher, and the loss hyperparameters transfer from Toys to Beauty and ML-1M without re-tuning.
- Against full-trajectory DreamRec, DimeRec and PDRec (NFE = 32), one-step RCCD is better on Toys, comparable on Beauty and slightly behind the best model on ML-1M, while being 32–157× faster; in quality-normalised cost (ms per HR point) it is more than an order of magnitude cheaper than all full-trajectory diffusion baselines.

### Citation

```bibtex
@mastersthesis{novikova2026diffusion,
  author = {Novikova, Alisa},
  title  = {Generative Diffusion Models for Recommendation},
  school = {HSE University, Faculty of Computer Science},
  year   = {2026},
  type   = {Master's thesis}
}
```

The citation of the DAMDID/RCDL 2026 paper will be added after publication.

---

<a id="русский"></a>
## Русский

### О работе

Репозиторий содержит код магистерской диссертации **«Генеративные диффузионные модели для рекомендательных систем»** (Новикова А. С., НИУ ВШЭ, факультет компьютерных наук, направление 01.04.02 «Прикладная математика и информатика», 2026; научный руководитель — Игнатов Д. И.).

По результатам этой диссертации подготовлена статья **«Ranking-Aligned Contrastive Consistency Distillation for One-Step Diffusion-Based Sequential Recommendation»** (А. Новикова, Д. И. Игнатов) — сокращённая версия работы, адаптированная под требования конференции. Статья принята на **DAMDID/RCDL 2026** — XXVIII Международную конференцию «Аналитика и управление данными в областях с интенсивным использованием данных» (Нижний Новгород, 20–23 октября 2026 г.), секция 6 «Recommender Systems and Scientific Knowledge Services».

> **Публикация:** _будет дополнено после конференции_ — издание, том, страницы, DOI.

### Мотивация

Диффузионные модели задают новый уровень качества последовательных рекомендаций, но платят за это итеративным инференсом: десятки шагов расшумления на запрос плохо совместимы с ограничениями по задержке в онлайн-сервисах. Простое усечение обратной траектории до одного шага не работает — учитель DiffuRec при NFE = 1 теряет примерно половину качества ранжирования.

### Цель

Обучить **одношагового ученика**, который воспроизводит качество ранжирования многошаговой латентно-диффузионной рекомендательной модели (DiffuRec) за один прямой проход.

Исследовательские вопросы:

- **RQ1.** Может ли consistency-дистилляция дать одношагового ученика, приближающегося по качеству к учителю с полной траекторией?
- **RQ2.** Улучшает ли выравнивание функции потерь с геометрией ранжирования (внутрибатчевый контрастный член) классическую consistency-дистилляцию, и статистически ли значим прирост?
- **RQ3.** Насколько устойчив ученик к бюджету инференса, датасету, случайному зерну, длине истории пользователя и популярности айтемов?

### Метод

RCCD объединяет три компонента:

1. **Дискретно-временная consistency-дистилляция** вдоль обратной траектории замороженного учителя DiffuRec (шаг учителя в стиле DDIM, EMA-копия ученика в качестве целевой сети).
2. **Residual-модуль уточнения** — лёгкий MLP поверх backbone'а и coarse-головы, унаследованных от учителя. Последний слой инициализирован нулями, поэтому в начале обучения ученик в точности совпадает с учителем.
3. **InfoNCE-член, выровненный с ранжированием** — внутрибатчевая контрастная функция потерь с якорем на эмбеддинге истинного следующего айтема, согласующая геометрию обучения с ранжированием по скалярному произведению по всему каталогу.

Архитектура ученика одинакова для всех вариантов функции потерь (CD-only и RCCD), поэтому основная абляция изолирует вклад именно функционала.

### Входные данные

Три открытых бенчмарка последовательных рекомендаций:

| Датасет | Пользователи | Айтемы | Взаимодействия | Ср. длина | Разреженность |
|---|---:|---:|---:|---:|---:|
| Amazon Beauty | 22 363 | 12 101 | 198 502 | 8,53 | 99,93% |
| Amazon Toys | 19 412 | 11 924 | 167 597 | 8,63 | 99,93% |
| MovieLens-1M | 6 040 | 3 416 | 999 611 | 165,50 | 95,16% |

Каждый датасет ожидается по пути `<data_root>/<dataset>/dataset.pkl` (имена: `toys`, `amazon_beauty`, `ml-1m`; по умолчанию `data_root` = `../datasets/data`). Pickle — словарь с ключами:

- `train` — `{user_id: [item_id, ...]}`, история в хронологическом порядке;
- `val` — `{user_id: [item_id]}`, предпоследнее взаимодействие;
- `test` — `{user_id: [item_id]}`, последнее взаимодействие;
- `smap` — отображение идентификаторов айтемов (его размер задаёт размер каталога).

Файлы `dataset.pkl` в репозиторий не входят из-за размера (см. `.gitignore`).

### Выходные данные

- чекпоинты учителя и ученика (`.pt`);
- логи обучения и CSV с метриками по эпохам;
- JSON с результатами по всем зернам и значениям NFE (`multiseed_results.json`), результаты подбора гиперпараметров;
- `test_predictions_nfe1.npz` с предсказаниями по пользователям для стратифицированного анализа;
- таблицы и графики диссертации, которые строит `analyzes.py`.

Метрики: HR@{5,10,20} и NDCG@{5,10,20} по полному каталогу (протокол leave-last-out), а также задержка на один пример в мс.

### Структура репозитория

```
original_diffurec/          учитель DiffuRec
  diffurec.py                 диффузионный процесс, расписания шума, денойзер
  model.py                    Att_Diffuse_model, функции потерь
  step_sample.py              сэмплеры шагов диффузии
  trainer.py                  обучение и оценка учителя
  utils.py                    загрузчики данных
  main.py                     точка входа для обучения учителя
consistency_diffurec/       ученик RCCD и эксперименты
  consistency_diffurec.py     ConsistencyStudent (backbone + модуль уточнения)
  distill_trainer.py          цикл дистилляции и оценка при заданном NFE
  evaluation.py               усечённый DDIM-учитель, замер задержки
  distill_main.py             точка входа: одиночный запуск / sweep / multi-seed
  hp_selection.py             подбор гиперпараметров на отборочных зернах
  multi_seed_runner.py        финальные запуски на оценочных зернах
  sanity_check.py             проверка переноса гиперпараметров между датасетами
  analyzes.py                 таблицы, статистические тесты и графики
```

### Запуск

```bash
pip install -r requirements.txt
export PYTHONPATH=original_diffurec:consistency_diffurec

# учитель
python original_diffurec/main.py --dataset toys --data_root ../datasets/data

# одиночный запуск RCCD (если --teacher_ckpt не задан, учитель обучается с нуля)
python consistency_diffurec/distill_main.py --dataset toys \
    --teacher_ckpt checkpoints/teacher_toys.pt \
    --contrast_weight 0.5 --contrast_temperature 0.1

# сетка по (beta, tau), включая базовый вариант CD-only
python consistency_diffurec/distill_main.py --dataset toys --sweep --include_baseline

# финальные запуски на пяти оценочных зернах вместе с CD-only
python consistency_diffurec/distill_main.py --dataset toys --multiseed --multiseed_baseline
```

Полный пайплайн диссертации: `hp_selection.py` (Toys, отборочные зерна {2017, 2024, 1997}) → `multi_seed_runner.py` с `--best_beta --best_tau --best_lr` (оценочные зерна {1907, 1977, 2015, 23, 88}) → `sanity_check.py` → `analyzes.py`. Эксперименты выполнялись в Google Colab на NVIDIA A100; `hp_selection.py`, `multi_seed_runner.py`, `sanity_check.py` и `analyzes.py` используют пути Google Drive из констант `DRIVE_BASE` / `*_ROOT` в начале файлов — для локального запуска их нужно поменять.

### Результаты

HR@10 (%) при NFE = 1, среднее по пяти зернам:

| | Toys | Beauty | ML-1M |
|---|---:|---:|---:|
| DiffuRec, полная траектория | 7,53 | 7,83 | 26,26 |
| DiffuRec, усечение до 1 шага | 3,67 | 4,08 | 13,42 |
| **RCCD, 1 шаг** | **8,06** | **8,15** | **26,52** |

Задержка на один пример (мс, batch 1, A100):

| | Toys | Beauty | ML-1M |
|---|---:|---:|---:|
| DiffuRec, полная траектория | 2,244 | 2,238 | 203,638 |
| **RCCD, 1 шаг** | **0,070** | **0,068** | **1,299** |
| Ускорение | 32,1× | 32,9× | 156,8× |

Основные выводы:

- Наивное усечение сохраняет лишь 48–52% HR@10 учителя; ученик RCCD восстанавливает 100,9–110,7% качества полной траектории по всем метрикам и датасетам.
- Добавление InfoNCE-члена к consistency-дистилляции даёт статистически значимое улучшение во всех 18 парных сравнениях по зернам после поправки Холма–Бонферрони; прирост максимален на больших отсечках и на айтемах из длинного хвоста.
- Качество ученика почти не зависит от NFE, разброс по зернам меньше, чем у усечённого учителя, а гиперпараметры функции потерь переносятся с Toys на Beauty и ML-1M без перенастройки.
- По сравнению с DreamRec, DimeRec и PDRec с полной траекторией (NFE = 32) одношаговый RCCD лучше на Toys, сопоставим на Beauty и немного уступает лучшей модели на ML-1M, будучи в 32–157 раз быстрее; по стоимости, нормированной на качество (мс на пункт HR), он более чем на порядок дешевле всех диффузионных бейзлайнов с полной траекторией.

### Цитирование

См. BibTeX в английском разделе. Ссылка на статью DAMDID/RCDL 2026 будет добавлена после публикации.
