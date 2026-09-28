"""Inférence NER PARHAF → CSV de soumission (sans liaison SNOMED).

BIO tokens → offsets caractères. À lancer une fois ; l'app Streamlit lit le CSV.

Usage :
  python infer_submission.py
  python infer_submission.py --split test
  python infer_submission.py --ckpt /chemin/vers/S1_18_score_0.7940
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

DEMO_DIR = Path(__file__).resolve().parent
REPO = DEMO_DIR.parents[1]
SNOBERT_SRC = REPO / "snobert" / "src"
sys.path.insert(0, str(SNOBERT_SRC))

from data_parhaf import PARHAF_INFECTIOLOGY_TYPES, build_bio_label2id  # noqa: E402
from model import CustomModel  # noqa: E402

_HOME = Path(os.environ["SNOMED_EL_HOME"]) if os.environ.get("SNOMED_EL_HOME") else REPO
DEFAULT_CKPT = (
    _HOME
    / "snobert"
    / "output_parhaf_ner"
    / "models"
    / "S1_18_score_0.7940"
)
DATA_DIR = Path(os.environ.get("PARHAF_DATA", REPO / "snobert" / "data" / "parhaf_infectiology"))
MAX_LEN = 512
FR_KEEP = re.compile(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿœŒæÆ\s.,:/\-']")


def preprocess_fr(text: str) -> str:
    return FR_KEEP.sub(" ", text)


def entity_type(tag: str) -> str:
    return tag.split("-", 1)[-1] if "-" in tag else tag


def join_bio(
    tokens: list[tuple[int, int, str, float]],
) -> list[tuple[int, int, str, float]]:
    """Une mention = plus long B-X suivi de I-X (même type).
    
        C'est le span BIO de référence, pas une ligne par sous-mot. CamemBERT peut
        étiqueter des morceaux du même mot en B-B-B avec écart 0 / chevauchement :
        ces morceaux sont collés. Un B-X plus loin après un espace démarre une nouvelle mention.
        """
    out: list[tuple[int, int, str, float]] = []
    i = 0
    n = len(tokens)
    while i < n:
        start, end, tag, conf = tokens[i]
        typ = entity_type(tag)
        if typ == "O":
            i += 1
            continue
        span_s, span_e = start, end
        confs = [conf]
        i += 1
        while i < n:
            start2, end2, tag2, conf2 = tokens[i]
            typ2 = entity_type(tag2)
            if typ2 != typ:
                break
            gap = start2 - span_e
            is_i = tag2.startswith("I-")
            is_b = tag2.startswith("B-")
            if is_i and gap <= 2:
                span_s = min(span_s, start2)
                span_e = max(span_e, end2)
                confs.append(conf2)
                i += 1
                continue
            if is_b and start2 <= span_e:
                span_s = min(span_s, start2)
                span_e = max(span_e, end2)
                confs.append(conf2)
                i += 1
                continue
            break
        out.append((span_s, span_e, typ, float(sum(confs) / len(confs))))
    return out


def load_model(ckpt: Path, num_classes: int, device: torch.device) -> tuple[CustomModel, object]:
    model = CustomModel(model_name=str(ckpt), fc_dropout=0, num_classes=num_classes)
    fc_state = torch.load(ckpt / "fc.pth", map_location=device)
    model.load_state_dict(fc_state, strict=False)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), use_fast=True)
    return model, tokenizer


def predict_note(
    model: CustomModel,
    tokenizer,
    note_id: str,
    raw_text: str,
    id2label: dict[int, str],
    device: torch.device,
) -> list[dict[str, object]]:
    text = preprocess_fr(raw_text)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    block = MAX_LEN - 2
    token_hits: list[tuple[int, int, str, float]] = []

    for i in range(0, len(input_ids), block):
        chunk_ids = [tokenizer.cls_token_id] + input_ids[i : i + block] + [tokenizer.sep_token_id]
        chunk_off = [(0, 0)] + offsets[i : i + block] + [(0, 0)]
        batch = {
            "input_ids": torch.tensor(chunk_ids, dtype=torch.long).unsqueeze(0).to(device),
            "attention_mask": torch.ones(1, len(chunk_ids), dtype=torch.long, device=device),
        }
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(**batch)[0]
            probs = torch.softmax(logits.float(), dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
        for off, pred_i, prob_row in zip(chunk_off[1:-1], pred_ids[1:-1], probs[1:-1]):
            start, end = int(off[0]), int(off[1])
            if end <= start:
                continue
            lab = id2label[int(pred_i.item())]
            if lab == "O":
                continue
            token_hits.append((start, end, lab, float(prob_row[int(pred_i.item())].item())))

    rows: list[dict[str, object]] = []
    for start, end, label, conf in join_bio(token_hits):
        span = raw_text[start:end]
        stripped = span.strip()
        if not stripped:
            continue
        lead = len(span) - len(span.lstrip())
        start = start + lead
        end = start + len(stripped)
        rows.append(
            {
                "note_id": note_id,
                "start": start,
                "end": end,
                "extracted_span": raw_text[start:end],
                "label": label,
                "confidence_score": round(conf, 6),
            }
        )
    return rows


def infer_split(
    notes_path: Path,
    ckpt: Path,
    out_path: Path,
) -> pd.DataFrame:
    notes = pd.read_csv(notes_path)
    label2id = build_bio_label2id(entity_types=PARHAF_INFECTIOLOGY_TYPES)
    id2label = {v: k for k, v in label2id.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(ckpt, num_classes=len(label2id), device=device)
    all_rows: list[dict[str, object]] = []
    for row in notes.itertuples(index=False):
        all_rows.extend(
            predict_note(
                model=model,
                tokenizer=tokenizer,
                note_id=str(row.note_id),
                raw_text=str(row.text),
                id2label=id2label,
                device=device,
            )
        )
    pred = pd.DataFrame(all_rows)
    if not pred.empty:
        pred = pred.sort_values(
            ["note_id", "start", "end"], kind="mergesort"
        ).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="PARHAF NER → submission CSV")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--split",
        choices=["val", "test", "both"],
        default="both",
        help="Which notes_*.csv to score (default: val and test)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEMO_DIR / "data")
    args = parser.parse_args()
    splits = ["val", "test"] if args.split == "both" else [args.split]
    parts: list[pd.DataFrame] = []
    for name in splits:
        notes_path = DATA_DIR / f"notes_{name}.csv"
        out_path = args.out_dir / f"submission_{name}.csv"
        print(f"infer {name}: {notes_path} -> {out_path}")
        pred = infer_split(notes_path=notes_path, ckpt=args.ckpt, out_path=out_path)
        pred.insert(0, "split", name)
        parts.append(pred)
        print(f"  spans={len(pred)}")
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(
        ["split", "note_id", "start", "end"], kind="mergesort"
    ).reset_index(drop=True)
    combined_path = args.out_dir / "submission.csv"
    combined.to_csv(combined_path, index=False)
    print(f"wrote {combined_path} rows={len(combined)} ckpt={args.ckpt}")


if __name__ == "__main__":
    main()
