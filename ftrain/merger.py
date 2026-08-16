
import os, sys, io, torch, math, re, gc
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
from typing import Optional, Dict, Any, List

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
                cap_cfg = TrainConfig(model_name=config.captain_model, captain_model=config.captain_model, captain_mode="llm", answer_mode="auto_yes")
                self.captain = PhoenixCaptain(cap_cfg)
            except: pass

    def _compute_loss(self, model, tokenizer, device) -> float:
        if not self.config.calibration_data: return float("inf")
        cal_data = load_data(self.config.calibration_data)[:10]
        ds = FtrainDataset(cal_data, tokenizer, 512)
        loader = DataLoader(ds, batch_size=2, collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id or 0))
        model.eval(); total, n = 0.0, 0
        with torch.no_grad():
            for b in loader:
                out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), labels=b["labels"].to(device))
                if out.loss is not None:
                    total += out.loss.item()
                    n += 1
        return total / max(1, n)

    def merge(self) -> bool:
        ui.fire_header()
        bar = ui.LoadingBar(message=f"Loading {self.model_a}"); bar.start()
        model1, tok = FastLanguageModel.from_pretrained(self.model_a, load_in_4bit=False, dtype=torch.float32); bar.done()
        
        bar = ui.LoadingBar(message=f"Loading {self.model_b}"); bar.start()
        model2, _ = FastLanguageModel.from_pretrained(self.model_b, load_in_4bit=False, dtype=torch.float32); bar.done()

        sd1, sd2 = model1.state_dict(), model2.state_dict()
        
        # ==========================================
        # VRAM OPTIMIZATION 1: Delete model2 early!
        # ==========================================
        del model2
        gc.collect()
        torch.cuda.empty_cache()

        keys = list(sd1.keys()); total = len(keys); merged = {}

        cal_loader = None
        if self.config.calibration_data and (self.config.use_fisher or self.config.repair_steps > 0 or self.config.merge_knowledge_distill or self.config.hugging):
            cal_data = load_data(self.config.calibration_data)
            cal_dataset = FtrainDataset(cal_data, tok, 512)
            cal_loader = DataLoader(cal_dataset, batch_size=4, collate_fn=partial(collate, pad_token_id=tok.pad_token_id or 0))

        if self.config.use_fisher:
            compatible = all(k in sd2 and sd1[k].shape == sd2[k].shape for k in sd1)
            if not compatible:
                print("⚠️ Models have different architectures. Disabling Fisher merge automatically (Incompatible shapes).")
                self.config.use_fisher = False

        fisher_a = fisher_b = None
        if self.config.use_fisher and cal_loader:
            print("🎣 Computing Fisher...")
            fisher_a = compute_fisher(model1, cal_loader, torch.device("cuda"))
            fisher_b = compute_fisher(model1, cal_loader, torch.device("cuda")) # Using model1 to avoid reloading model2
            
            if fisher_a is None or fisher_b is None:
                print("⚠️ Falling back to Intelligent SLERP/Weighted merging without Fisher.")
                self.config.use_fisher = False

        analyzer, planner = MergeAnalyzer(), MergePlanner()
        stages = [(0,0.25,"Analyzing embeddings"), (0.25,0.5,"Merging attention"), (0.5,0.75,"Merging FFN"), (0.75,1.0,"Safety check")]

        for i, k in enumerate(keys):
            progress = i / total; msg = ""
            for s, e, t in stages:
                if s <= progress < e: msg = t; break
            if i % max(1, total//20) == 0 or i == total-1: ui.print_merge_progress(i+1, total, message=msg)

            if k not in sd2 or sd1[k].shape != sd2[k].shape:
                merged[k] = sd1[k]; continue
            a, b = sd1[k], sd2[k]

            if self.strategy == "weighted": merged[k] = fast_weighted_avg(a, b, self.alpha)
            elif self.strategy == "fisher" and self.config.use_fisher and fisher_a and k in fisher_a: 
                merged[k] = fast_fisher_merge(a, b, fisher_a[k], fisher_b[k])
            elif self.strategy == "slerp": merged[k] = fast_slerp(a, b, self.alpha)
            elif self.strategy == "ties": merged[k] = fast_ties(a, b)
            elif self.strategy == "intelligent":
                analysis = analyzer.analyze_pair(k, a, b)
                plan = planner.plan_for_pair(k, analysis)
                if self.captain and i == 0:
                    cap_plan = self.captain.inspect_merge({"name": self.model_a}, {"name": self.model_b}, analysis)
                    print(f"🧠 Captain merge plan: {cap_plan}")
                    
                    # ==========================================
                    # VRAM OPTIMIZATION 2: Delete Captain LLM!
                    # ==========================================
                    del self.captain
                    gc.collect()
                    torch.cuda.empty_cache()

                if plan.strategy == "keep_a": merged[k] = a
                elif plan.strategy == "keep_b": merged[k] = b
                elif plan.strategy == "weighted": merged[k] = fast_weighted_avg(a, b, plan.alpha)
                elif plan.strategy == "slerp": merged[k] = fast_slerp(a, b, plan.alpha)
                elif plan.strategy == "ties": merged[k] = fast_ties(a, b)
                elif plan.strategy == "projection":
                    if a.dim() == 2:
                        from .projection import apply_projection
                        P = apply_projection(a, plan.projection, b)
                        proj_a = (a.float() @ P).to(a.dtype)
                        merged[k] = (plan.alpha * proj_a.float() + (1 - plan.alpha) * b.float()).to(a.dtype)
                    else:
                        merged[k] = (plan.alpha * a.float() + (1 - plan.alpha) * b.float()).to(a.dtype)
                else: merged[k] = a
            else: merged[k] = (self.alpha * a.float() + (1 - self.alpha) * b.float()).to(a.dtype)

        # ==========================================
        # VRAM OPTIMIZATION 3: Delete sd2 BEFORE Safety Check!
        # ==========================================
        del sd2
        gc.collect()
        torch.cuda.empty_cache()

        rep = check_state_dict(merged, baseline=sd1, norm_collapse_factor=0.1)
        print("\n" + rep.summary())
        if not rep.ok: merged = sanitize(merged, sd1, norm_collapse_factor=0.1)
        
        # ==========================================
        # VRAM OPTIMIZATION 4: Delete sd1 BEFORE loading merged weights!
        # ==========================================
        del sd1
        gc.collect()
        torch.cuda.empty_cache()

        model1.load_state_dict(merged)
        model1 = model1.to(self.dtype)

        if self.config.name == "auto":
            a_name = self.model_a.split("/")[-1].replace("-", "_")
            b_name = self.model_b.split("/")[-1].replace("-", "_")
            repo_name = f"{a_name}_{b_name}_IntelMerge"
        else:
            repo_name = self.config.name
            
        os.makedirs(self.output_dir, exist_ok=True)
        model1.save_pretrained(self.output_dir)
        tok.save_pretrained(self.output_dir)
        
        hf_url = "None"
        if self.config.hugging and self.config.hugging_token:
            print("📊 Benchmarking models for HuggingFace push...")
            device = torch.device("cuda")
            loss_merged = self._compute_loss(model1, tok, device)
            
            try:
                m_a, t_a = FastLanguageModel.from_pretrained(self.model_a, load_in_4bit=True, dtype=torch.float16)
                loss_a_orig = self._compute_loss(m_a, t_a, device)
                del m_a, t_a; gc.collect(); torch.cuda.empty_cache()
            except: loss_a_orig = float('inf')
            
            try:
                m_b, t_b = FastLanguageModel.from_pretrained(self.model_b, load_in_4bit=True, dtype=torch.float16)
                loss_b_orig = self._compute_loss(m_b, t_b, device)
                del m_b, t_b; gc.collect(); torch.cuda.empty_cache()
            except: loss_b_orig = float('inf')

            avg_orig = (loss_a_orig + loss_b_orig) / 2
            print(f"Loss A: {loss_a_orig:.4f} | Loss B: {loss_b_orig:.4f} | Avg: {avg_orig:.4f} | Merged: {loss_merged:.4f}")
            
            if loss_merged < avg_orig:
                print("✅ Merged model is smarter than average! Pushing to HuggingFace...")
                from huggingface_hub import HfApi
                api = HfApi(token=self.config.hugging_token)
                if "/" not in repo_name:
                    user = api.whoami()
                    if "name" in user: repo_name = f"{user['name']}/{repo_name}"
                api.create_repo(repo_id=repo_name, token=self.config.hugging_token, exist_ok=True, repo_type="model")
                api.upload_folder(folder_path=self.output_dir, repo_id=repo_name, token=self.config.hugging_token, repo_type="model")
                hf_url = f"https://huggingface.co/{repo_name}"
                print(f"🚀 Pushed to HuggingFace: {hf_url}")
            else:
                print("❌ Merged model is dumber than average. Skipping HuggingFace push.")

        stats = {"Model A": self.model_a, "Model B": self.model_b, "Output Dir": self.output_dir, "HF Repo": hf_url}
        ui.print_final_summary(stats)
        return True
