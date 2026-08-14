import re, threading, time, json, torch, numpy as np
from collections import deque
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

class PhoenixCaptain:
    def __init__(self, config, answer_mode=None):
        self.config = config
        self.async_mode = config.captain_mode not in ("rule", "disabled")
        self.mode = config.captain_mode
        self.previous_loss = None
        self._last_result = None
        self._last_applied = None
        self._busy = False
        self._lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self._last_call_ts = 0.0
        self.is_moe = False
        self.expert_imbalance = None
        self.model_profile = None
        self.data_profile = None
        self.answer_mode = answer_mode or getattr(config, "answer_mode", "auto_yes")
        self.memory = deque(maxlen=10)
        self.loss_history = deque(maxlen=getattr(config, "captain_velocity_window", 3))
        self.reward_history = deque(maxlen=3)

        if self.mode == "llm" and getattr(config, "captain_model", None):
            try:
                from unsloth import FastLanguageModel
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.captain_model,
                    max_seq_length=2048,
                    load_in_4bit=True,
                    dtype=torch.float16
                )
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("🧠 Captain LLM loaded via Unsloth.")
            except Exception as e:
                logger.warning(f"⚠️ Captain LLM failed to load ({str(e)[:50]}...). Falling back to rule-based mode.")
                self.mode = "rule"

    def set_family_context(self, family, is_moe):
        self.is_moe = is_moe

    def update_expert_imbalance(self, imb):
        self.expert_imbalance = imb

    def analyze_model(self, model):
        p = {}
        try:
            c = model.config
            p["model_type"] = getattr(c, "model_type", "unknown")
            p["num_layers"] = getattr(c, "num_hidden_layers", 0)
            p["hidden_size"] = getattr(c, "hidden_size", 0)
            from .model_utils import count_params
            p["param_stats"] = count_params(model)
        except:
            pass
        self.model_profile = p

    def analyze_data(self, dataset, tokenizer, max_samples=200):
        lengths = []
        for i in range(min(len(dataset), max_samples)):
            it = dataset[i]
            ids = it.get("input_ids")
            if ids is None:
                ids = tokenizer.encode(it.get("text", ""), add_special_tokens=False)
            lengths.append(len(ids))
        if lengths:
            self.data_profile = {"avg_length": float(np.mean(lengths)), "p95": float(np.percentile(lengths, 95))}

    def analyze_and_report_data(self, orig_len, new_len, changes):
        report = f"🧹 Analyzed {orig_len} samples.\n"
        if orig_len == new_len and not changes:
            report += "✅ Data quality is excellent. No anomalies detected."
        else:
            report += "🔧 Actions taken:\n"
            for change in changes:
                report += f"  - {change}\n"
            report += f"📊 Final size: {new_len} samples (Removed {orig_len - new_len})."
        if self.model and self.mode == "llm":
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_inference(self.model)
                inp = self.tokenizer(f"Summarize this data report in 1 sentence:\n{report}", return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(**inp, max_new_tokens=50, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                report += f"\n🧠 Captain's Verdict: {self.tokenizer.decode(out[0][inp['input_ids'].shape[-1]:], skip_special_tokens=True).strip()}"
            except:
                pass
        from . import ui
        ui.print_captain_report(report)

    def evaluate_improvement(self, prompt, before_text, after_text, correct_text):
        report = f"📊 Pre/Post Training Evaluation\n\n❓ Prompt: {prompt[:100]}...\n\n❌ Before: {before_text.strip()[:150]}...\n✅ After:  {after_text.strip()[:150]}...\n🎯 Expected: {correct_text.strip()[:150]}...\n\n"
        if self.model and self.mode == "llm":
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_inference(self.model)
                inp = self.tokenizer(f"Compare Before and After to Expected. Give % improvement and 1 sentence why.\n{report}", return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(**inp, max_new_tokens=100, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                report += f"🧠 Evaluation: {self.tokenizer.decode(out[0][inp['input_ids'].shape[-1]:], skip_special_tokens=True).strip()}"
            except:
                report += "🧠 Evaluation: LLM Error."
        else:
            if correct_text.strip() in after_text.strip() and correct_text.strip() not in before_text.strip():
                report += "🧠 Evaluation: 100% improvement!"
            else:
                report += "🧠 Evaluation: Model logic shifted, manual review required."
        from . import ui
        ui.print_captain_report(report)

    def _rule_advice(self, step, loss, lr, grad_norm, brain_regions, val_loss=None):
        early, late, gate = brain_regions
        trend = "stable"
        if grad_norm < 1e-6:
            return {"message": "Gradient collapse", "action": "Boost LR", "mult": 2.0, "layer_boost": "all", "stop": False}
        self.loss_history.append(loss)
        if len(self.loss_history) >= 3:
            accel = (self.loss_history[-1] - self.loss_history[-2]) - (self.loss_history[-2] - self.loss_history[-3])
            if accel > 0.05:
                return {"message": "Loss accelerating", "action": "Aggressive LR Cut", "mult": 0.4, "layer_boost": "none", "stop": False}
            if np.var(np.array(self.loss_history)) < 1e-4:
                return {"message": "Plateau detected", "action": "Boost LR", "mult": 1.5, "layer_boost": "all", "stop": False}
        if self.previous_loss is not None:
            diff = loss - self.previous_loss
            if diff > 0.05:
                trend = "rising"
            elif diff < -0.05:
                trend = "falling"
        self.previous_loss = loss
        if val_loss is not None and len(self.memory) > 2:
            pv = [m.get("val_loss") for m in self.memory if "val_loss" in m]
            if pv and val_loss > np.mean(pv[-3:]):
                return {"message": "Val loss rising", "action": "Decrease LR", "mult": 0.7, "layer_boost": "none", "stop": False}
        if self.is_moe and self.expert_imbalance and self.expert_imbalance > 0.6:
            return {"message": "Expert imbalance", "action": "Decrease LR", "mult": 0.7, "layer_boost": "none", "stop": False}
        elif late > 0.2 and early < 0.05:
            return {"message": "Superficial learning", "action": "Deep Brain Stimulus", "mult": 1.5, "layer_boost": "early", "stop": False}
        elif early > 0.2 and late < 0.05:
            return {"message": "Perception overload", "action": "Cortex Stimulus", "mult": 1.5, "layer_boost": "late", "stop": False}
        elif trend == "rising" and grad_norm > 5.:
            return {"message": "Instability", "action": "Decrease LR", "mult": 0.5, "layer_boost": "none", "stop": False}
        return {"message": "Stable", "action": "Keep LR", "mult": 1.0, "layer_boost": "none", "stop": False}

    def inspect_training(self, step, loss, lr, grad_norm, brain_regions, val_loss=None):
        rule_res = self._rule_advice(step, loss, lr, grad_norm, brain_regions, val_loss)
        if self.model and self.mode == "llm":
            now = time.time()
            if now - self._last_call_ts >= self.config.captain_min_interval:
                self._last_call_ts = now
                if self.async_mode:
                    with self._lock:
                        if self._busy:
                            return rule_res
                        self._busy = True
                    def _worker():
                        try:
                            from unsloth import FastLanguageModel
                            FastLanguageModel.for_inference(self.model)
                            prompt = f"Step {step}, loss {loss:.4f}, lr {lr:.2e}, grad_norm {grad_norm:.4f}. Advice?"
                            inp = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
                            with torch.no_grad():
                                out = self.model.generate(**inp, max_new_tokens=50, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                            res = {"message": self.tokenizer.decode(out[0][inp['input_ids'].shape[-1]:], skip_special_tokens=True).strip(), "action": "LLM Advise", "mult": 1.0, "layer_boost": "none", "stop": False}
                            with self._lock:
                                self._last_result = res
                        except:
                            pass
                        finally:
                            with self._lock:
                                self._busy = False
                    threading.Thread(target=_worker, daemon=True).start()
        with self._lock:
            self._last_result = rule_res
        self.memory.append({"step": step, "loss": loss, "advice": rule_res.get("action", "")})
        return rule_res

    def get_latest_advice(self):
        with self._lock:
            if self._last_result and self._last_result != self._last_applied:
                self._last_applied = self._last_result
                return self._last_result
        return None
