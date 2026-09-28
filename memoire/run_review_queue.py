from __future__ import annotations

"""Construit la file de relecture à partir de l'alignement hold-out."""
import pandas as pd

from .paths import ALIGN_A, EXAMPLES_CSV, OUTPUT_DIR, QUEUE_CSV, QUEUE_MD
from .review_queue import add_queue_flag, queue_by_regime, queue_metrics, sample_examples


def _markdown(metrics: dict[str, float], by_regime: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# File de relecture (option A, sans gold)",
            "",
            "Politique EDS : on **émet** KIRIs ∪ SnoBERT ; on **relit** tout sauf ACCORD.",
            "L'oracle n'est pas utilisé à l'inférence : seulement `regime` (désaccord / unilatéral).",
            "",
            "## Table 3 — efficacité de la file",
            "",
            f"| Mentions | {int(metrics['n_mentions'])} |",
            f"| Erreurs (ni KIRIs ni SnoBERT justes) | {int(metrics['n_errors'])} |",
            f"| Taille de la file | {int(metrics['n_queue'])} "
            f"({100 * metrics['queue_coverage']:.1f} % des mentions) |",
            f"| Rappel des erreurs (file) | {100 * metrics['error_recall']:.1f} % |",
            f"| Précision des erreurs (file) | {100 * metrics['error_precision']:.1f} % |",
            f"| Erreurs restantes dans ACCORD (file aveugle) | "
            f"{int(metrics['n_errors_in_accord'])} "
            f"({100 * metrics['accord_error_share']:.1f} % des erreurs) |",
            "",
            "## Détail par régime",
            "",
            "| Régime | Dans la file ? | n | Erreurs | P(erreur) | Part des erreurs |",
            "|---|---|---:|---:|---:|---:|",
            *[
                f"| {row.regime} | {'oui' if row.in_queue else 'non'} | {int(row.n)} | "
                f"{int(row.n_errors)} | {row.p_error:.3f} | {100 * row.share_of_all_errors:.1f} % |"
                for row in by_regime.itertuples(index=False)
            ],
            "",
        ]
    )


def run(align_path=ALIGN_A) -> dict[str, float]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aligned = pd.read_csv(align_path)
    flagged = add_queue_flag(aligned)
    flagged.to_csv(QUEUE_CSV, index=False)
    metrics = queue_metrics(aligned)
    by_regime = queue_by_regime(aligned)
    QUEUE_MD.write_text(_markdown(metrics, by_regime), encoding="utf-8")
    sample_examples(aligned, per_regime=2).to_csv(EXAMPLES_CSV, index=False)
    return metrics


if __name__ == "__main__":
    scores = run()
    for key, value in scores.items():
        print(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
