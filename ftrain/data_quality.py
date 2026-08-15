import math, json, hashlib, numpy as np, torch

def compute_perplexity(text, model, tok, dev, ml=512):
    inp = tok(text, return_tensors="pt", truncation=True, max_length=ml).to(dev)
    if inp["input_ids"].numel() == 0:
        return float('inf')
    with torch.no_grad():
        l = model(**inp, labels=inp["input_ids"]).loss.item()
    return math.exp(l) if l < 50 else float('inf')

def filter_by_perplexity(data, model, tok, dev, keep_pct=0.8, ml=512):
    c = [x.get("text") if isinstance(x.get("text"), str) else str(x.get("messages","")) for x in data]
    s = [compute_perplexity(t, model, tok, dev, ml) for t in c]
    th = np.quantile(s, keep_pct)
    return [x for x, sc in zip(data, s) if sc <= th]

def deduplicate(data):
    seen, u = set(), []
    for x in data:
        h = hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()
        if h not in seen:
            seen.add(h)
            u.append(x)
    return u

def balance_datasets(sources, strategy="tokens"):
    if not sources:
        return []
    if len(sources) == 1:
        return sources[0]
    if strategy == "samples":
        m = max(len(s) for s in sources)
        b = []
        for s in sources:
            if s:
                b.extend(s * max(1, m // len(s)))
        return b
    tt = [sum(len(str(x.get("text", x.get("messages", "")))) for x in s) for s in sources]
    mt = max(tt) if tt else 0
    b = []
    for s, t in zip(sources, tt):
        if t > 0 and s:
            b.extend(s * max(1, int(mt / t)))
    return b
