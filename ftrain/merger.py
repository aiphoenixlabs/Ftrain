
import os, sys, io, torch, math, re, gc
import torch.nn.functional as F
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
from typing import Optional, Dict, Any, List, Tuple

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

    def _purge_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _compute_loss(self, model, tokenizer, device) -> float:
        if not self.config.calibration_data: return float("inf")
        cal_data = load_data(self.config.calibration_data)[:10]
        ds = FtrainDataset(cal_data, tokenizer, 512)
        loader = DataLoader(ds, batch_size=2, collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id or 0))
        model.eval(); total, n = 0.0, 0
        with torch.inference_mode():
            for b in loader:
                out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), labels=b["labels"].to(device))
                if out.loss is not None:
                    total += out.loss.item()
                    n += 1
        return total / max(1, n)

    def _get_num_layers(self, state_dict: Dict[str, torch.Tensor]) -> int:
        layers = set()
        for k in state_dict.keys():
            match = re.search(r'(?:layers|h)\.(\d+)\.', k)
            if match: layers.add(int(match.group(1)))
        return max(layers) + 1 if layers else 1

    def _find_matching_key(self, key_a: str, keys_b: List[str], num_layers_a: int, num_layers_b: int) -> Optional[str]:
        if key_a in keys_b: return key_a
        layer_match = re.search(r'(?:model\.layers|decoder\.layers|transformer\.h)\.(\d+)\.', key_a)
        layer_a = int(layer_match.group(1)) if layer_match else None
        layer_b = layer_a
        if layer_a is not None and num_layers_a > 1 and num_layers_b > 1:
            layer_b = int(round(layer_a * (num_layers_b - 1) / (num_layers_a - 1)))
        norm_key = key_a
        if layer_a is not None:
            norm_key = re.sub(r'(?:model\.layers|decoder\.layers|transformer\.h)\.\d+\.', f'model.layers.{layer_b}.', key_a)
        aliases = [
            ('self_attn.q_proj', 'attention.wq'), ('self_attn.k_proj', 'attention.wk'),
            ('self_attn.v_proj', 'attention.wv'), ('self_attn.o_proj', 'attention.wo'),
            ('mlp.gate_proj', 'feed_forward.w1'), ('mlp.up_proj', 'feed_forward.w3'),
            ('mlp.down_proj', 'feed_forward.w2'), ('input_layernorm', 'attention_norm'),
            ('post_attention_layernorm', 'ffn_norm')
        ]
        candidates = [norm_key]
        for src, target in aliases:
            if src in norm_key: candidates.append(norm_key.replace(src, target))
            elif target in norm_key: candidates.append(norm_key.replace(target, src))
        for cand in candidates:
            if cand in keys_b: return cand
        return None

    def _align_tensor_shapes(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape == b.shape: return b
        
        # ==========================================
        # VRAM OPTIMIZATION: Do interpolation on CPU RAM!
        # ==========================================
        device = a.device
        a_cpu = a.detach().to("cpu", dtype=torch.float32)
        b_cpu = b.detach().to("cpu", dtype=torch.float32)
        
        a_dim, b_dim = a_cpu.dim(), b_cpu.dim()

        if a_dim == 1 and b_dim == 1:
            b_reshaped = b_cpu.unsqueeze(0).unsqueeze(0)
            b_aligned = F.interpolate(b_reshaped, size=(a_cpu.shape[0],), mode='linear', align_corners=False)
            return b_aligned.squeeze(0).squeeze(0).to(a.dtype).to(device)

        elif a_dim == 2 and b_dim == 2:
            b_reshaped = b_cpu.unsqueeze(0).unsqueeze(0)
            b_aligned = F.interpolate(b_reshaped, size=(a_cpu.shape[0], a_cpu.shape[1]), mode='bilinear', align_corners=False)
            return b_aligned.squeeze(0).squeeze(0).to(a.dtype).to(device)

        else:
            out = torch.zeros_like(a_cpu, dtype=torch.float32)
            slices = tuple(slice(0, min(sa, sb)) for sa, sb in zip(a_cpu.shape, b_cpu.shape))
            out[slices] = b_cpu[slices]
            return out.to(a.dtype).to(device)

    def merge(self) -> bool:
        ui.fire_header()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # STEP 1: Fisher Computation
        fisher_a, fisher_b = None, None
        cal_loader = None
        if self.config.calibration_data and (self.config.use_fisher or self.config.repair_steps > 0 or self.config.merge_knowledge_distill or self.config.hugging):
            cal_data = load_data(self.config.calibration_data)
            _, tok_temp = FastLanguageModel.from_pretrained(self.model_a, load_in_4bit=True)
            cal_dataset = FtrainDataset(cal_data, tok_temp, 512)
            cal_loader = DataLoader(cal_dataset, batch_size=2, collate_fn=partial(collate, pad_token_id=tok_temp.pad_token_id or 0))
            del tok_temp; self._purge_memory()

        if self.config.use_fisher and cal_loader:
            print("🎣 Computing Fisher for Model A...")
            m1_temp, _ = FastLanguageModel.from_pretrained(self.model_a, load_in_4bit=False, dtype=torch.float16)
            fisher_a = compute_fisher(m1_temp, cal_loader, device)
            del m1_temp; self._purge_memory()
            print("🎣 Computing Fisher for Model B...")
            m2_temp, _ = FastLanguageModel.from_pretrained(self.model_b, load_in_4bit=False, dtype=torch.float16)
            fisher_b = compute_fisher(m2_temp, cal_loader, device)
            del m2_temp; self._purge_memory()

        # STEP 2: Extract State Dicts to CPU & DELETE MODELS FROM VRAM
        bar = ui.LoadingBar(message=f"Loading {self.model_a} (CPU State Dict)"); bar.start()
        model1, tok = FastLanguageModel.from_pretrained(self.model_a, load_in_4bit=False, dtype=torch.float16)
        sd1 = {k: v.cpu() for k, v in model1.state_dict().items()}
        del model1; self._purge_memory()  # DELETE MODEL1 FROM VRAM
        bar.done()

        bar = ui.LoadingBar(message=f"Loading {self.model_b} (CPU State Dict)"); bar.start()
        model2, _ = FastLanguageModel.from_pretrained(self.model_b, load_in_4bit=False, dtype=torch.float16)
        sd2 = {k: v.cpu() for k, v in model2.state_dict().items()}
        del model2; self._purge_memory()  # DELETE MODEL2 FROM VRAM
        bar.done()

        num_layers_a = self._get_num_layers(sd1)
        num_layers_b = self._get_num_layers(sd2)
        keys_b = list(sd2.keys())
        sd1_baseline = {k: v.clone().to(torch.float16) for k, v in sd1.items()}

        # STEP 3: Merge Loop
        analyzer, planner = MergeAnalyzer(), MergePlanner()
        keys_a = list(sd1.keys())
        total = len(keys_a)
        stages = [(0, 0.25, "Analyzing embeddings"), (0.25, 0.5, "Merging attention"), (0.5, 0.75, "Merging FFN"), (0.75, 1.0, "Safety check")]

        for i, k_a in enumerate(keys_a):
            progress = i / total; msg = ""
            for s, e, t in stages:
                if s <= progress < e: msg = t; break
            if i % max(1, total // 20) == 0 or i == total - 1:
                ui.print_merge_progress(i + 1, total, message=msg)

            k_b = self._find_matching_key(k_a, keys_b, num_layers_a, num_layers_b)
            if not k_b: continue

            a = sd1[k_a].to(device, dtype=torch.float32)
            b_raw = sd2.pop(k_b).to(device, dtype=torch.float32)
            b = self._align_tensor_shapes(a, b_raw)
            del b_raw

            merged_tensor = None
            if self.strategy == "weighted":
                merged_tensor = fast_weighted_avg(a, b, self.alpha)
            elif self.strategy == "fisher" and fisher_a and fisher_b and k_a in fisher_a and k_b in fisher_b:
                fa = fisher_a[k_a].to(device)
                fb = fisher_b[k_b].to(device)
                merged_tensor = fast_fisher_merge(a, b, fa, fb)
            elif self.strategy == "slerp":
                merged_tensor = fast_slerp(a, b, self.alpha)
            elif self.strategy == "ties":
                merged_tensor = fast_ties(a, b)
            elif self.strategy == "intelligent":
                analysis = analyzer.analyze_pair(k_a, a, b)
                plan = planner.plan_for_pair(k_a, analysis)

                if self.captain and i == 0:
                    cap_plan = self.captain.inspect_merge({"name": self.model_a}, {"name": self.model_b}, analysis)
                    print(f"\n🧠 Captain merge plan: {cap_plan}")
                    del self.captain; self.captain = None; self._purge_memory()

                if plan.strategy == "keep_a": merged_tensor = a
                elif plan.strategy == "keep_b": merged_tensor = b
                elif plan.strategy == "weighted": merged_tensor = fast_weighted_avg(a, b, plan.alpha)
                elif plan.strategy == "slerp": merged_tensor = fast_slerp(a, b, plan.alpha)
                elif plan.strategy == "ties": merged_tensor = fast_ties(a, b)
                elif plan.strategy == "projection":
                    if a.dim() == 2:
                        from .projection import apply_projection
                        P = apply_projection(a, plan.projection, b)
                        proj_a = (a @ P)
                        merged_tensor = (plan.alpha * proj_a + (1 - plan.alpha) * b)
                    else:
                        merged_tensor = (plan.alpha * a + (1 - plan.alpha) * b)
                else:
                    merged_tensor = a
            else:
                merged_tensor = (self.alpha * a + (1 - self.alpha) * b)

            sd1[k_a] = merged_tensor.to(dtype=self.dtype, device="cpu")
            del a, b, merged_tensor

        del sd2; self._purge_memory()

        # STEP 4: Safety Check
        rep = check_state_dict(sd1, baseline=sd1_baseline, norm_collapse_factor=0.1)
        print("\n" + rep.summary())
        if not rep.ok:
            sd1 = sanitize(sd1, sd1_baseline, norm_collapse_factor=0.1)
        del sd1_baseline; self._purge_memory()

        # STEP 5: Reload Model1 to save
        print("💾 Reloading base model architecture for saving...")
        model1, tok = FastLanguageModel.from_pretrained(self.model_a, load_in_4bit=False, dtype=self.dtype)
        model1.load_state_dict(sd1)
        del sd1; self._purge_memory()

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
            loss_merged = self._compute_loss(model1, tok, device)
            try:
                m_a, t_a = FastLanguageModel.from_pretrained(self.model_a, load_in_4bit=True)
                loss_a_orig = self._compute_loss(m_a, t_a, device)
                del m_a, t_a; self._purge_memory()
            except: loss_a_orig = float('inf')
            try:
                m_b, t_b = FastLanguageModel.from_pretrained(self.model_b, load_in_4bit=True)
                loss_b_orig = self._compute_loss(m_b, t_b, device)
                del m_b, t_b; self._purge_memory()
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
