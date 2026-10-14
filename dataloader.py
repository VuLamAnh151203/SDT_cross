"""Read the original SDT IEMOCAP/MELD feature pickles without copying data."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


BASE_DIR = Path(__file__).resolve().parent


def resolve_feature_path(dataset, feature_path=None):
    if feature_path is not None:
        path = Path(feature_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("Feature pickle not found: {}".format(path))
        return path
    filename = dataset.lower() + "_multimodal_features.pkl"
    candidates = [BASE_DIR / "data" / filename, BASE_DIR.parent / "SDT" / "data" / filename]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Place {} in {} or pass --feature-path".format(filename, BASE_DIR / "data"))


class DialogueDataset(Dataset):
    def __init__(self, dataset="IEMOCAP", feature_path=None):
        if dataset not in ("IEMOCAP", "MELD"):
            raise ValueError("dataset must be IEMOCAP or MELD")
        self.dataset = dataset
        self.feature_path = resolve_feature_path(dataset, feature_path)
        with self.feature_path.open("rb") as stream:
            values = pickle.load(stream, encoding="latin1")
        expected = 12 if dataset == "IEMOCAP" else 13
        if len(values) != expected:
            raise ValueError("expected {} entries in the {} SDT pickle".format(expected, dataset))
        (self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,
         self.roberta2, self.roberta3, self.roberta4, self.videoAudio,
         self.videoVisual, self.videoSentence, train_vid, test_vid) = values[:12]
        self.trainVid, self.testVid = list(train_vid), list(test_vid)
        self.keys = self.trainVid + self.testVid
        if not self.trainVid or not self.testVid or len(set(self.keys)) != len(self.keys):
            raise ValueError("trainVid/testVid must be nonempty, unique and disjoint")
        self.n_classes = 6 if dataset == "IEMOCAP" else 7
        self.n_speakers = 2 if dataset == "IEMOCAP" else 9
        first = self.keys[0]
        self.feature_dims = {
            "D_text": np.asarray(self.videoText[first]).shape[-1],
            "D_visual": np.asarray(self.videoVisual[first]).shape[-1],
            "D_audio": np.asarray(self.videoAudio[first]).shape[-1],
        }
        for key in self.keys:
            labels = np.asarray(self.videoLabels[key])
            if labels.ndim != 1 or not len(labels) or labels.min() < 0 or labels.max() >= self.n_classes:
                raise ValueError("invalid labels in dialogue {}".format(key))
            for field, dimension in ((self.videoText, "D_text"), (self.videoVisual, "D_visual"),
                                     (self.videoAudio, "D_audio")):
                if np.asarray(field[key]).shape != (len(labels), self.feature_dims[dimension]):
                    raise ValueError("feature/label shape mismatch in {} ({})".format(key, dimension))
            if len(self.videoSpeakers[key]) != len(labels):
                raise ValueError("speaker/label length mismatch in {}".format(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        speakers = self.videoSpeakers[key]
        if self.dataset == "IEMOCAP":
            speakers = [[1, 0] if speaker == "M" else [0, 1] for speaker in speakers]
        return (
            torch.tensor(np.asarray(self.videoText[key]), dtype=torch.float32),
            torch.tensor(np.asarray(self.videoVisual[key]), dtype=torch.float32),
            torch.tensor(np.asarray(self.videoAudio[key]), dtype=torch.float32),
            torch.tensor(np.asarray(speakers), dtype=torch.float32),
            torch.ones(len(self.videoLabels[key])),
            torch.tensor(self.videoLabels[key], dtype=torch.long),
            key,
        )


def collate_dialogues(items):
    columns = list(zip(*items))
    return [pad_sequence(list(values), batch_first=index in (4, 5),
                         padding_value=-100 if index == 5 else 0)
            for index, values in enumerate(columns[:6])] + [list(columns[6])]


def split_dialogues(dataset, selection_protocol="test", valid_ratio=0.1):
    if selection_protocol == "test":
        # Same membership as SDT/train.py with valid=0.0.
        valid_ids, train_ids = [], dataset.trainVid
    elif selection_protocol == "validation":
        if not 0 < valid_ratio < 1:
            raise ValueError("valid_ratio must be between 0 and 1")
        count = int(valid_ratio * len(dataset.trainVid))
        if not 0 < count < len(dataset.trainVid):
            raise ValueError("validation ratio gives an empty train or validation split")
        valid_ids, train_ids = dataset.trainVid[:count], dataset.trainVid[count:]
    else:
        raise ValueError("selection_protocol must be test or validation")
    return {"train": list(train_ids), "valid": list(valid_ids), "test": list(dataset.testVid)}


def make_loaders(dataset, split_ids, batch_size, seed, num_workers=0, pin_memory=False):
    lookup = {key: index for index, key in enumerate(dataset.keys)}
    loaders = {}
    for split, keys in split_ids.items():
        if not keys:
            loaders[split] = None
            continue
        loaders[split] = DataLoader(
            Subset(dataset, [lookup[key] for key in keys]), batch_size=batch_size,
            shuffle=split == "train", collate_fn=collate_dialogues,
            generator=torch.Generator().manual_seed(seed), num_workers=num_workers,
            pin_memory=pin_memory)
    return loaders
