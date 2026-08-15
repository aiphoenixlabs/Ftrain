import random, torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Any, Iterator

class FtrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, use_packing=False):
        self.tok = tokenizer
        self.max_len = max_length
        ex = []
        for e in data:
            ids = self._enc(e)
            if ids and len(ids["input_ids"]) > 0:
                ex.append(ids)
        if use_packing:
            pk = self._pack([r["input_ids"] for r in ex], max_length)
            ex = [{"input_ids": p, "labels": list(p)} for p in pk]
        self.examples = ex
        self.lengths = [len(e["input_ids"]) for e in ex]
        if not ex:
            raise ValueError("Dataset empty!")

    def _enc(self, e):
        m = e.get("messages")
        if m:
            try:
                t = self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            except:
                t = "\n".join(f"{x.get('role','user')}: {x.get('content','')}" for x in m)
        else:
            t = e.get("text", "")
        if not isinstance(t, str):
            t = str(t)
        ids = self.tok(t, add_special_tokens=True, truncation=True, max_length=self.max_len)["input_ids"]
        eos = self.tok.eos_token_id
        if eos is not None and (not ids or ids[-1] != eos):
            ids.append(eos)
        return {"input_ids": ids[:self.max_len], "labels": list(ids[:self.max_len])}

    def _pack(self, s, ml):
        pk, buf, cur = [], [], 0
        for i in s:
            if cur + len(i) > ml and buf:
                pk.append(buf)
                buf, cur = [], 0
            buf.extend(i)
            cur += len(i)
        if buf:
            pk.append(buf)
        return pk

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        return {
            "input_ids": torch.tensor(e["input_ids"], dtype=torch.long),
            "attention_mask": torch.ones(len(e["input_ids"]), dtype=torch.long),
            "labels": torch.tensor(e["labels"], dtype=torch.long)
        }

def collate(batch, pad_token_id=0):
    out = {}
    for k in ("input_ids", "attention_mask", "labels"):
        pv = 0 if k == "attention_mask" else (-100 if k == "labels" else pad_token_id)
        out[k] = torch.nn.utils.rnn.pad_sequence([i[k] for i in batch], batch_first=True, padding_value=pv)
    return out

class LengthSampler(Sampler):
    def __init__(self, lengths, bs, shuffle=True, seed=42):
        self.lengths = lengths
        self.bs = bs
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        mega = self.bs * 50
        batches = []
        for i in range(0, len(idx), mega):
            c = idx[i:i+mega]
            c.sort(key=lambda x: self.lengths[x])
            for j in range(0, len(c), self.bs):
                batches.append(c[j:j+self.bs])
        if self.shuffle:
            rng.shuffle(batches)
        for b in batches:
            for i in b:
                yield i

    def __len__(self):
        return len(self.lengths)
