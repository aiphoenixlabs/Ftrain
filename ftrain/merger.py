import os, sys, io, torch, math
from functools import partial
from unsloth import FastLanguageModel
from . import ui
from .merge_intel import MergeAnalyzer, MergePlanner
from .safety import check_state_dict, sanitize
from .captain import PhoenixCaptain
from .merge_advanced import compute_fisher
from .data_utils import load_data
from .dataset import FtrainDataset, collate
from torch.utils.data import DataLoader
from .cpp_merge import fast_weighted_avg, fast_slerp, fast_ties, fast_fisher_merge

class Merger:
    def __init__(self, config):
        self.config = config
        self.model_a = config.model_a
        self.model_b = config.model_b
        self.output_dir = config.output_dir
        self.strategy = config.strategy
        self.alpha = config.alpha
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config.save_dtype]
        self.captain = None
        if config.captain_model:
            try:
                from .config import TrainConfig
                self.captain = PhoenixCaptain(TrainConfig(model_name=config.captain_model, captain_model=config.captain_model, captain_mode="llm", answer_mode="auto_yes"))
            except:
                pass

    def _compute_loss(self, m, tok, dev):
        if not self.config.calibration_data:
            return float("inf")
        ds = FtrainDataset(load_data(self.config.calibration_data)[:10], tok, 512)
        ld = DataLoader(ds, batch_size=2, collate_fn=partial(collate, pad_token_id=tok.pad_token_id or 0))
        m.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in ld:
                out = m(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev), labels=b["labels"].to(dev))
                tot += out.loss.item()
                n += 1
        return tot / max(1, n)

    def merge(self):
        ui.fire_header()
        bar = ui.LoadingBar(message=f"Loading {self.model_a}")
        bar.start()
        m1, tok = FastLanguageModel.from_pretrained(self.model_a, load_in_4bit=False, dtype=torch.float32)
        bar.done()
        bar = ui.LoadingBar(message=f"Loading {self.model_b}")
        bar.start()
        m2, _ = FastLanguageModel.from_pretrained(self.model_b, load_in_4bit=False, dtype=torch.float32)
        bar.done()

        sd1, sd2 = m1.state_dict(), m2.state_dict()
        keys = list(sd1.keys())
        tot = len(keys)
        merged = {}

        cal = None
        if self.config.calibration_data and (self.config.use_fisher or self.config.repair_steps > 0 or self.config.merge_knowledge_distill or self.config.hugging):
            cd = load_data(self.config.calibration_data)
            cds = FtrainDataset(cd, tok, 512)
            cal = DataLoader(cds, batch_size=4, collate_fn=partial(collate, pad_token_id=tok.pad_token_id or 0))

        fa = fb = None
        if self.config.use_fisher and cal:
            print("🎣 Computing Fisher...")
            fa = compute_fisher(m1, cal, torch.device("cuda"))
            fb = compute_fisher(m2, cal, torch.device("cuda"))

        an, pl = MergeAnalyzer(), MergePlanner()
        st = [(0, 0.25, "Analyzing embeddings"), (0.25, 0.5, "Merging attention"), (0.5, 0.75, "Merging FFN"), (0.75, 1.0, "Safety check")]

        for i, k in enumerate(keys):
            pr = i / tot
            msg = ""
            for s, e, t in st:
                if s <= pr < e:
                    msg = t
                    break
            if i % max(1, tot // 20) == 0 or i == tot - 1:
                ui.print_merge_progress(i + 1, tot, msg)
            if k not in sd2 or sd1[k].shape != sd2[k].shape:
                merged[k] = sd1[k]
                continue
            a, b = sd1[k], sd2[k]
            if self.strategy == "weighted":
                merged[k] = fast_weighted_avg(a, b, self.alpha)
            elif self.strategy == "fisher" and fa and k in fa:
                merged[k] = fast_fisher_merge(a, b, fa[k], fb[k])
            elif self.strategy == "slerp":
                merged[k] = fast_slerp(a, b, self.alpha)
            elif self.strategy == "ties":
                merged[k] = fast_ties(a, b)
            elif self.strategy == "intelligent":
                al = an.analyze_pair(k, a, b)
                pn = pl.plan_for_pair(k, al)
                if self.captain and i == 0:
                    print(f"🧠 Captain merge plan: {self.captain.inspect_merge({'name': self.model_a}, {'name': self.model_b}, al)}")
                if pn.strategy == "keep_a":
                    merged[k] = a
                elif pn.strategy == "keep_b":
                    merged[k] = b
                elif pn.strategy == "weighted":
                    merged[k] = fast_weighted_avg(a, b, pn.alpha)
                elif pn.strategy == "slerp":
                    merged[k] = fast_slerp(a, b, pn.alpha)
                elif pn.strategy == "ties":
                    merged[k] = fast_ties(a, b)
                elif pn.strategy == "projection":
                    if a.dim() == 2:
                        from .projection import apply_projection
                        P = apply_projection(a, pn.projection, b)
                        pa = (a.float() @ P).to(a.dtype)
                        merged[k] = (pn.alpha * pa.float() + (1 - pn.alpha) * b.float()).to(a.dtype)
                    else:
                        merged[k] = (pn.alpha * a.float() + (1 - pn.alpha) * b.float()).to(a.dtype)
                else:
                    merged[k] = a
            else:
                merged[k] = (self.alpha * a.float() + (1 - self.alpha) * b.float()).to(a.dtype)

        rep = check_state_dict(merged, sd1, 0.1)
        print("\n" + rep.summary())
        if not rep.ok:
            merged = sanitize(merged, sd1, 0.1)

        m1.load_state_dict(merged)
        m1 = m1.to(self.dtype)

        if self.config.repair_steps > 0 and cal:
            print(f"🔧 Repair fine-tuning {self.config.repair_steps} steps...")
            ro = torch.optim.AdamW(m1.parameters(), lr=1e-5)
            m1.train()
            for stp, b in enumerate(cal):
                if stp >= self.config.repair_steps:
                    break
                b = {k: v.to("cuda") for k, v in b.items()}
                l = m1(**b).loss
                l.backward()
                ro.step()
                ro.zero_grad()
                print(f"   Repair step {stp + 1}/{self.config.repair_steps}, loss {l.item():.4f}")

        if self.config.name == "auto":
            rn = f"{self.model_a.split('/')[-1].replace('-', '_')}_{self.model_b.split('/')[-1].replace('-', '_')}_IntelMerge"
        else:
            rn = self.config.name

        os.makedirs(self.output_dir, exist_ok=True)
        m1.save_pretrained(self.output_dir)
        tok.save_pretrained(self.output_dir)

        hf = "None"
        if self.config.hugging and self.config.hugging_token:
            print("📊 Benchmarking models for HuggingFace push...")
            dev = torch.device("cuda")
            lm = self._compute_loss(m1, tok, dev)
            la = self._compute_loss(FastLanguageModel.from_pretrained(self.model_a, load_in_4bit=True, dtype=torch.float16)[0], tok, dev)
            lb = self._compute_loss(FastLanguageModel.from_pretrained(self.model_b, load_in_4bit=True, dtype=torch.float16)[0], tok, dev)
            avg = (la + lb) / 2
            print(f"Loss A: {la:.4f} | Loss B: {lb:.4f} | Avg: {avg:.4f} | Merged: {lm:.4f}")
            if lm < avg:
                print("✅ Merged model is smarter than average! Pushing to HuggingFace...")
                from huggingface_hub import HfApi
                api = HfApi(token=self.config.hugging_token)
                if "/" not in rn:
                    u = api.whoami()
                    if "name" in u:
                        rn = f"{u['name']}/{rn}"
                api.create_repo(repo_id=rn, token=self.config.hugging_token, exist_ok=True, repo_type="model")
                api.upload_folder(folder_path=self.output_dir, repo_id=rn, token=self.config.hugging_token, repo_type="model")
                hf = f"https://huggingface.co/{rn}"
                print(f"🚀 Pushed to HuggingFace: {hf}")
            else:
                print("❌ Merged model is dumber than average. Skipping HuggingFace push.")

        ui.print_final_summary({"Model A": self.model_a, "Model B": self.model_b, "Output Dir": self.output_dir, "HF Repo": hf})
        return True
