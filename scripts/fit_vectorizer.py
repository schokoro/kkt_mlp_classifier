"""Standalone fit TF-IDF: Parquet с колонкой ``name`` → автономный артефакт векторизатора.

Метки не нужны; внешние сервисы и предобученные веса не используются.
Чтение корпуса ограничено --max-rows (по умолчанию 1 000 000): файл читается
по row-группам и весь в память не загружается. Ограничение воспроизводимо:
берутся первые max_rows непустых name в порядке следования строк файла
(стратегия «head»); результат зависит от версии файла и порядка строк.
Для случайной выборки заранее подготовьте перемешанный сэмпл.

Пример:
    python scripts/fit_vectorizer.py --input corpus.parquet --output outputs/tfidf_corpus \\
        --max-rows 1000000 --features-preset default --source-id corpus_v1
    python scripts/run_train.py --preset default --override \\
        '{"features": {"mode": "pretrained", "pretrained_path": "outputs/tfidf_corpus"},
          "data": {"path": "labeled.parquet"}}'
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf import build_feature_config, fit_and_save_vectorizer, setup_logging  # noqa: E402
from kkt_clf.vectorizer import DEFAULT_CORPUS_MAX_ROWS  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Parquet с колонкой name (метки не требуются)")
    p.add_argument("--output", required=True, help="Каталог автономного артефакта векторизатора")
    p.add_argument("--max-rows", type=int, default=DEFAULT_CORPUS_MAX_ROWS,
                   help=f"Верхний предел строк, поступающих в fit (по умолчанию {DEFAULT_CORPUS_MAX_ROWS}); "
                        "берутся первые непустые name в порядке следования. 0 или меньше — ошибка; "
                        "чтобы читать файл целиком, задайте заведомо большое число")
    p.add_argument("--features-preset", default="default", help="default | ods_light (источник конфига TF-IDF)")
    p.add_argument("--override", default="{}",
                   help="JSON с переопределениями FeatureConfig (ветки char/word, vectorizer_fit_rows)")
    p.add_argument("--seed", type=int, default=42, help="seed подвыборки vectorizer_fit_rows")
    p.add_argument("--source-id", default=None, help="Идентификатор/версия подготовленного корпуса (в метаданные)")
    p.add_argument("--force", action="store_true", help="Разрешить перезапись непустого каталога --output")
    a = p.parse_args()
    setup_logging()
    feat_cfg = build_feature_config(a.features_preset, json.loads(a.override))
    _, meta = fit_and_save_vectorizer(
        a.output, feat_cfg, a.input,
        max_rows=a.max_rows, seed=a.seed, source_id=a.source_id, overwrite=a.force,
    )
    src = meta["source"]
    print(json.dumps({
        "artifact": os.path.abspath(a.output),
        "source": src["path"],
        "source_id": src["source_id"],
        "rows_read": src["rows_read"],
        "rows_dropped_empty": src["rows_dropped_empty"],
        "rows_used": src["rows_used"],
        "rows_fit": meta["fit"]["rows_fit"],
        "fit_subsampled": meta["fit"]["subsampled"],
        "truncated": src["truncated"],
        "vocab": meta["vocab"],
        "seed": meta["seed"],
    }, ensure_ascii=False, indent=2))
