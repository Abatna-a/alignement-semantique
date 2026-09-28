"""Jeux de données et labels BIO pour l'entraînement PARHAF."""
import bisect
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_sctid_syn(sctid_syn_parh: Path):
    with open(sctid_syn_parh, "r") as f:
        data = json.load(f)
        return set(map(int, data.keys()))


SNOMED_BIO_TYPES: tuple[str, ...] = ("find", "proc", "body")
PARHAF_INFECTIOLOGY_TYPES: tuple[str, ...] = (
    "Infection",
    "Site",
    "Bacterie",
    "Bacteriemie",
)


def build_bio_label2id(entity_types: tuple[str, ...]) -> dict[str, int]:
    """Identifiants BIO : O=0, puis B-type / I-type pour chaque type d'entité dans l'ordre."""
    mapping: dict[str, int] = {"O": 0}
    next_id = 1
    for entity_type in entity_types:
        mapping[f"B-{entity_type}"] = next_id
        mapping[f"I-{entity_type}"] = next_id + 1
        next_id += 2
    return mapping


def default_snomed_label2id() -> dict[str, int]:
    return build_bio_label2id(entity_types=SNOMED_BIO_TYPES)


def add_parhaf_class(ann_df: pd.DataFrame) -> pd.DataFrame:
    """Utilise les chaînes concept_id PARHAF comme types NER (sans dictionnaire de synonymes SNOMED)."""
    out = ann_df.copy()
    out["cls"] = out["concept_id"].astype(str).str.strip()
    unknown = sorted(set(out["cls"]) - set(PARHAF_INFECTIOLOGY_TYPES))
    if unknown:
        n_before = len(out)
        out = out.loc[out["cls"].isin(PARHAF_INFECTIOLOGY_TYPES)].copy()
        print(
            f"Warning: dropping {n_before - len(out)} PARHAF rows with "
            f"unexpected concept_id {unknown}"
        )
    return out


def add_concept_class(ann_df, sctid_syn_dir: Path):
    # Identifiants racine pour Procedure / Clinical finding / Body structure
    ProcId, FindId, BodyId = 71388002, 404684003, 123037004
    p_cids = load_sctid_syn(Path(sctid_syn_dir) / "proc_sctid_syn.json")
    p_cids.add(ProcId)
    f_cids = load_sctid_syn(Path(sctid_syn_dir) / "find_sctid_syn.json")
    f_cids.add(FindId)
    b_cids = load_sctid_syn(Path(sctid_syn_dir) / "body_sctid_syn.json")
    b_cids.add(BodyId)
    snomed_class = []
    keep_idx = []
    unknown_cids = set()
    for i, r in ann_df.iterrows():
        cid = int(r.concept_id)
        if cid in p_cids:
            label = "proc"
        elif cid in b_cids:
            label = "body"
        elif cid in f_cids:
            label = "find"
        else:
            unknown_cids.add(cid)
            continue
        snomed_class.append(label)
        keep_idx.append(i)

    if unknown_cids:
        n_drop = len(ann_df) - len(keep_idx)
        print(
            f"Warning: skipping {n_drop} annotation rows with "
            f"{len(unknown_cids)} unknown concept_ids "
            f"(not in body/find/proc syn dicts; often due to SNOMED version mismatch)."
        )

    ann_df = ann_df.loc[keep_idx].copy()
    ann_df["cls"] = snomed_class
    return ann_df


def get_labels(starts, ends, spans):
    """Convertit les offsets en étiquettes de séquence au format BIO."""
    labels = ["O"] * len(starts)
    spans = sorted(spans)
    for s, e, l in spans:
        li = bisect.bisect_left(starts, s)
        ri = bisect.bisect_left(starts, e)
        ni = len(labels[li:ri])
        labels[li] = f"B-{l}"
        labels[li + 1 : ri] = [f"I-{l}"] * (ni - 1)
    return labels


class Labeler:
    def __init__(self, tokenizer, ascii_only: bool = True):
        self.tokenizer = tokenizer
        self.ascii_only = ascii_only

    def fix_annotation(self, text, ann_df):
        trail = False
        idxs = []
        for i, c in enumerate(text):
            if c == " ":
                if trail:
                    idxs.append(i)
                trail = True
            else:
                trail = False

        label_char = np.zeros(len(text))
        label_char[idxs] = 1
        label_char = np.cumsum(label_char).astype(int)

        char_spans = []
        for i, r in ann_df.iterrows():
            span = [r.start, r.end, r.cls]
            char_spans.append(span)

        # Corriger les spans, retirer les espaces en fin de texte
        for i, span in enumerate(char_spans):
            s, e, c = span
            s -= label_char[s]
            e -= label_char[e]
            char_spans[i] = [s, e, c]
        return char_spans

    def preprocess_text(self, note_ann_df, text):
        if self.ascii_only:
            text = re.sub(r"[^a-zA-Z0-9\s.,:\/]", " ", text)
        else:
            # Garder les lettres françaises ; remplacer les autres symboles par un espace (même longueur).
            text = re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿœŒæÆ\s.,:\/\-']", " ", text)
        char_spans = []
        for _, r in note_ann_df.iterrows():
            span = [r.start, r.end, r.cls]
            char_spans.append(span)

        encoded = self.tokenizer(
            text, add_special_tokens=False, truncation=False, return_offsets_mapping=True
        )
        input_ids = encoded["input_ids"]
        off = np.array(encoded["offset_mapping"])
        starts = off[:, 0]
        ends = off[:, 1]

        labels = get_labels(starts, ends, char_spans)
        return text, input_ids, labels


def convert_labels_tokens(tokenizer, note_df, ann_df, ascii_only: bool = True):
    preproc = Labeler(tokenizer, ascii_only=ascii_only)

    agg = ann_df.groupby("note_id")
    # Vers un dictionnaire
    anns = {}
    for note_id, r in tqdm(agg):
        anns[note_id] = r

    res = defaultdict(list)
    for i, r in tqdm(note_df.iterrows(), total=len(note_df)):
        note_id = r.note_id
        text = r.text
        ann_note_df = anns[note_id]
        try:
            t, i, l = preproc.preprocess_text(ann_note_df, text)
            res["note_id"].append(note_id)
            res["text"].append(t)
            res["input_ids"].append(i)
            res["labels"].append(l)
            res["fold"].append(r.fold)
        except Exception as e:
            print(e)
            print(note_id)

    tdf = pd.DataFrame(res)
    return tdf


def parallel_convert_labels_tokens(
    tokenizer, note_df, ann_df, n_jobs=4, ascii_only: bool = True
):
    from joblib import Parallel, delayed

    # pandas>=2.2/3 : np.array_split(DataFrame) renvoie des ndarrays, pas des DataFrames
    note_df = note_df.reset_index(drop=True)
    idx_chunks = np.array_split(np.arange(len(note_df)), n_jobs)
    note_chunks = [note_df.iloc[idx] for idx in idx_chunks]
    ann_df_gg = ann_df.groupby("note_id")
    ann_df_dict = {k: v for k, v in ann_df_gg}
    ann_chunks = []
    for chunk in note_chunks:
        note_ids = chunk.note_id
        ann_chunk = pd.concat([ann_df_dict[nid] for nid in note_ids])
        ann_chunks.append(ann_chunk)

    res = Parallel(n_jobs=n_jobs)(
        delayed(convert_labels_tokens)(
            tokenizer, note_chunk, ann_chunk, ascii_only=ascii_only
        )
        for note_chunk, ann_chunk in zip(note_chunks, ann_chunks)
    )
    tdf = pd.concat(res)
    return tdf


class PreprocessedDataset(Dataset):
    def __init__(
        self,
        cfg,
        tokenizer,
        df,
        train=False,
        fold=None,
        repeat=1,
        feature=None,
        label2id: dict[str, int] | None = None,
    ):
        self.cfg = cfg
        if fold is not None:
            if isinstance(fold, int):
                df = df[df.fold == fold]
            elif isinstance(fold, list):
                df = df[df.fold.isin(fold)]

        self.df = df
        self.tokenizer = tokenizer
        self.repeat = repeat
        self.train = train
        self.feature = feature

        self.label2id = (
            default_snomed_label2id() if label2id is None else dict(label2id)
        )
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.note_dict = df.set_index("note_id").to_dict()["text"]

        if not self.train:
            # Couper input_ids et label_str en 2, puis concaténer
            res = []
            for i, r in df.iterrows():
                input_ids = r.input_ids
                label_str = r.labels
                input_ids1 = input_ids[: len(input_ids) // 2]
                label_str1 = label_str[: len(label_str) // 2]
                input_ids2 = input_ids[len(input_ids) // 2 :]
                label_str2 = label_str[len(label_str) // 2 :]
                row1 = [r.note_id, r.text, input_ids1, label_str1, r.fold]
                row2 = [r.note_id, r.text, input_ids2, label_str2, r.fold]
                res.append(row1)
                res.append(row2)
            self.df = pd.DataFrame(res)
            self.df.columns = ["note_id", "text", "input_ids", "labels", "fold"]
        print("Dataset len:", len(self.df))
        print("Average token len:", self.df.input_ids.apply(len).mean())

    def __len__(self):
        return len(self.df) * self.repeat

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise StopIteration
        idx = idx % len(self.df)

        _, text, input_ids, label_str, _ = self.df.iloc[idx]

        max_len = self.cfg.max_len - 2

        if len(input_ids) > max_len:
            if self.train:
                offset_fixed = np.random.randint(0, len(input_ids) - max_len)
            else:
                offset_fixed = 0

            input_ids = input_ids[offset_fixed : offset_fixed + max_len]
            label_str = label_str[offset_fixed : offset_fixed + max_len]
        else:
            offset_fixed = 0

        labels_int = [self.label2id[l] for l in label_str]
        labels_int = [-100] + labels_int + [-100]
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels_int, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class ChunkedDataset:
    def __init__(self, tokenizer, fold, df, max_len, repeat=1, label2id=None):
        self.label2id = (
            default_snomed_label2id() if label2id is None else dict(label2id)
        )
        self.id2label = {v: k for k, v in self.label2id.items()}
        if fold is not None:
            if isinstance(fold, int):
                df = df[df.fold == fold]
            elif isinstance(fold, list):
                df = df[df.fold.isin(fold)]

        _max_len = max_len - 2
        # Découper les notes en chunks de max_len
        chunked_rows = []
        for i, row in df.iterrows():
            ids = row["input_ids"]
            labels = row["labels"]

            for i in range(0, len(ids), _max_len):
                chunked_rows.append(
                    {
                        "fold": row["fold"],
                        "ids": ids[i : i + _max_len],
                        "labels": labels[i : i + _max_len],
                    }
                )
        df = pd.DataFrame(chunked_rows)
        print(f"chunked into {len(df)} rows")
        self.df = df
        self.tokenizer = tokenizer
        self.repeat = repeat
        self.max_len = max_len

    def __len__(self):
        return len(self.df) * self.repeat

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise StopIteration
        idx = idx % len(self.df)

        row = self.df.iloc[idx]
        input_ids = row["ids"]
        label_str = row["labels"]

        labels_int = [self.label2id[l] for l in label_str]
        labels_int = [-100] + labels_int + [-100]
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels_int, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
