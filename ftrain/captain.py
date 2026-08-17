

import re, threading, time, json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

class PhoenixCaptain:
    def __init__(self, config, answer_mode: Optional[str] = None):
        self.config = config
        self.async_mode = config.captain_mode not in ("rule", "disabled")
        self.mode = config.captain_mode
        self.previous_loss = None; self._last_result = None; self._last_applied = None
        self._busy = False; self._lock = threading.Lock()
        self.model = None; self.tokenizer = None; self._last_call_ts = 0.0
        self.is_moe = False; self.expert_imbalance = None
        self.model_profile = None; self.data_profile = None
        self.answer_mode = answer_mode or getattr(config, "answer_mode", "auto_yes")
        self.memory = deque(maxlen=10)
        self.loss_history = deque(maxlen=getattr(config, "captain_velocity_window", 3))
        self.reward_history = deque(maxlen=3)
        
        if self.mode == "llm" and getattr(config, "captain_model", None):
            try:
                from unsloth import FastLanguageModel
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.captain_model, max_seq_length=2048, load_in_4bit=True, dtype=torch.float16
                )
                if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("🧠 Captain LLM loaded via Unsloth.")
            except Exception as e:
                logger.warning(f"⚠️ Captain LLM failed to load ({str(e)[:50]}...). Falling back to rule-based mode.")
                self.mode = "rule"

    def set_family_context(self, family: str, is_moe: bool): 
        self.is_moe = is_moe
        
    def update_expert_imbalance(self, imb: float): 
        self.expert_imbalance = imb
    
    def analyze_model(self, model: Any):
        p = {}
        try:
            c = model.config
            p["model_type"] = getattr(c, "model_type", "unknown")
            p["num_layers"] = getattr(c, "num_hidden_layers", 0)
            p["hidden_size"] = getattr(c, "hidden_size", 0)
            from .model_utils import count_params
            p["param_stats"] = count_params(model)
        except: pass
        self.model_profile = p

    def analyze_and_report_data(self, orig_len: int, new_len: int, changes: List[str]):
        """Generates a beautiful report of data quality and changes made."""
        report = f"🧹 Analyzed {orig_len} samples.\n"
        if orig_len == new_len and not changes:
            report += "✅ Data quality is excellent. No anomalies detected. Proceeding with raw dataset."
        else:
            report += "🔧 Actions taken to clean data:\n"
            for change in changes: report += f"  - {change}\n"
            report += f"📊 Final dataset size: {new_len} samples (Removed {orig_len - new_len})."
        
        if self.model and self.mode == "llm":
            prompt = f"You are a data quality expert. Summarize this data cleaning report in one sentence:\n{report}"
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_inference(self.model)
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
                with torch.no_grad(): out = self.model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                llm_summary = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                report += f"\n\n🧠 Captain's Verdict: {llm_summary.strip()}"
            except: pass
        from . import ui
        ui.print_captain_report(report)

    def evaluate_improvement(self, prompt: str, before_text: str, after_text: str, correct_text: str) -> str:
        """Evaluates the model's intelligence improvement before and after training."""
        report = f"📊 Pre/Post Training Evaluation\n\n"
        report += f"❓ Prompt: {prompt[:100]}...\n\n"
        report += f"❌ Before Training: {before_text.strip()[:150]}...\n"
        report += f"✅ After Training:  {after_text.strip()[:150]}...\n"
        report += f"🎯 Expected:        {correct_text.strip()[:150]}...\n\n"
        
        if self.model and self.mode == "llm":
            eval_prompt = f"You are an AI evaluator. Compare the 'Before' and 'After' responses to the 'Expected' response. Give a percentage of how much the model improved.\n{report}\nImprovement Score (e.g., '45%') and 1 sentence why:"
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_inference(self.model)
                inputs = self.tokenizer(eval_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
                with torch.no_grad(): out = self.model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                llm_eval = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                report += f"🧠 Captain's Evaluation: {llm_eval.strip()}"
            except:
                report += "🧠 Captain's Evaluation: Evaluation failed (LLM error)."
        else:
            if correct_text.strip() in after_text.strip() and correct_text.strip() not in before_text.strip():
                report += "🧠 Captain's Evaluation: 100% improvement! Model successfully learned the target logic."
            else:
                report += "🧠 Captain's Evaluation: Model logic shifted, but manual review required."
        
        from . import ui
        ui.print_captain_report(report)

    def analyze_data(self, dataset: Any, tokenizer: Any, max_samples: int = 200):
        lengths = []
        for i in range(min(len(dataset), max_samples)):
            it = dataset[i]; ids = it.get("input_ids")
            if ids is None:
                text = it.get("text", "")
                if not isinstance(text, str): text = it.get("content", "")
                if not isinstance(text, str): text = ""
                ids = tokenizer.encode(text, add_special_tokens=False)
            lengths.append(len(ids))
        if lengths: self.data_profile = {"avg_length": float(np.mean(lengths)), "p95": float(np.percentile(lengths, 95))}

    def _rule_advice(self, step: int, loss: float, lr: float, grad_norm: float, brain_regions: Tuple[float, float, float], val_loss: Optional[float] = None) -> Dict[str, Any]:
        early, late, gate = brain_regions; trend = "stable"
        if grad_norm < 1e-6: return {"message": "Gradient collapse detected", "action": "Boost LR & Unfreeze", "mult": 2.0, "layer_boost": "all", "stop": False}
        self.loss_history.append(loss)
        if len(self.loss_history) >= 3:
            accel = (self.loss_history[-1] - self.loss_history[-2]) - (self.loss_history[-2] - self.loss_history[-3])
            if accel > 0.05: return {"message": "Loss acceleration detected", "action": "Aggressive LR Cut", "mult": 0.4, "layer_boost": "none", "stop": False}
            loss_arr = np.array(self.loss_history)
            if np.var(loss_arr) < 1e-4: return {"message": "Training plateau detected", "action": "Boost LR", "mult": 1.5, "layer_boost": "all", "stop": False}

        if self.previous_loss is not None:
            diff = loss - self.previous_loss
            if diff > 0.05: trend = "rising"
            elif diff < -0.05: trend = "falling"
        self.previous_loss = loss
        
        if val_loss is not None and len(self.memory) > 2:
            pv = [m.get("val_loss") for m in self.memory if "val_loss" in m]
            if pv and val_loss > np.mean(pv[-3:]): return {"message": "Validation loss increasing", "action": "Decrease LR", "mult": 0.7, "layer_boost": "none", "stop": False}
        if self.is_moe and self.expert_imbalance and self.expert_imbalance > 0.6: return {"message": "Expert imbalance high", "action": "Decrease LR", "mult": 0.7, "layer_boost": "none", "stop": False}
        elif late > 0.2 and early < 0.05: return {"message": "Superficial learning", "action": "Deep Brain Stimulus", "mult": 1.5, "layer_boost": "early", "stop": False}
        elif early > 0.2 and late < 0.05: return {"message": "Perception overload", "action": "Cortex Stimulus", "mult": 1.5, "layer_boost": "late", "stop": False}
        elif trend == "rising" and grad_norm > 5.: return {"message": "Brain instability", "action": "Decrease LR", "mult": 0.5, "layer_boost": "none", "stop": False}
        return {"message": "Stable training", "action": "Keep LR", "mult": 1.0, "layer_boost": "none", "stop": False}

    def _llm_parse(self, response: str) -> Dict[str, Any]:
        diag = re.search(r"Diagnosis:\s*(.+?)(?:\n|Action:)", response, re.I | re.S)
        act = re.search(r"Action:\s*(.+?)(?:\n|Multiplier:|$)", response, re.I | re.S)
        mult = re.search(r"Multiplier:\s*([0-9.]+)", response, re.I)
        lo, hi = self.config.captain_clamp; mval = max(lo, min(hi, float(mult.group(1)) if mult else 1.0))
        return {"message": diag.group(1).strip() if diag else "LLM advice", "action": act.group(1).strip() if act else "Keep LR", "mult": mval, "layer_boost": "none", "stop": False}

    def _build_prompt(self, step: int, loss: float, lr: float, grad_norm: float, brain_regions: Tuple[float, float, float], val_loss: Optional[float] = None) -> str:
        system = "You are a deep-learning training advisor."
        mi = json.dumps(self.model_profile) if self.model_profile else ""
        di = json.dumps(self.data_profile) if self.data_profile else ""
        early, late, gate = brain_regions
        hist = "\n".join([f"Step {m['step']}: loss {m['loss']:.4f}, advice '{m['advice']}'" for m in self.memory])
        return f"""{system}\nModel: {mi}\nData: {di}\nStep {step}, loss {loss:.4f}, lr {lr:.2e}, grad_norm {grad_norm:.4f}, early_grad {early:.3f}, late_grad {late:.3f}, gate_grad {gate:.3f}. Val loss: {val_loss if val_loss else 'N/A'}.\nHistory: {hist}\nProvide Diagnosis, Action, Multiplier (0.25-2.5)."""

    def inspect_training(self, step: int, loss: float, lr: float, grad_norm: float, brain_regions: Tuple[float, float, float], val_loss: Optional[float] = None) -> Dict[str, Any]:
        rule_res = self._rule_advice(step, loss, lr, grad_norm, brain_regions, val_loss)
        if self.model and self.mode == "llm":
            now = time.time()
            if now - self._last_call_ts >= self.config.captain_min_interval:
                self._last_call_ts = now; prompt = self._build_prompt(step, loss, lr, grad_norm, brain_regions, val_loss)
                if self.async_mode:
                    with self._lock:
                        if self._busy: return rule_res
                        self._busy = True
                    def _worker():
                        try:
                            from unsloth import FastLanguageModel
                            FastLanguageModel.for_inference(self.model)
                            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
                            with torch.no_grad():
                                out = self.model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.2, pad_token_id=self.tokenizer.eos_token_id)
                            res = self._llm_parse(self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
                            with self._lock: self._last_result = res
                        except Exception as e:
                            logger.warning(f"⚠️ Captain LLM generation failed ({str(e)[:50]}...). Falling back to rules.")
                            self.mode = "rule"
                        finally:
                            with self._lock: self._busy = False
                    threading.Thread(target=_worker, daemon=True).start()
        with self._lock: self._last_result = rule_res
        self.memory.append({"step": step, "loss": loss, "advice": self._last_result.get("action", "") if self._last_result else ""})
        return rule_res

    def inspect_merge(self, info1: Dict[str, Any], info2: Dict[str, Any], tensor_analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """Inspects tensor statistics for model merging and provides strategy recommendations."""
        sim = tensor_analysis.get("similarity", 0.5) if tensor_analysis else 0.5
        cat = tensor_analysis.get("category", "other") if tensor_analysis else "other"

        if sim < 0.3 and cat not in ("norm", "router"):
            return {
                "action": "keep_a",
                "alpha": 1.0,
                "reason": f"Low similarity ({sim:.2f}) in {cat} tensor. Retaining primary model weights."
            }
        elif sim > 0.85:
            return {
                "action": "weighted_average",
                "alpha": 0.5,
                "reason": f"High similarity ({sim:.2f}) in {cat} tensor. Standard linear blending safe."
            }
        else:
            return {
                "action": "slerp",
                "alpha": 0.5,
                "reason": f"Moderate similarity ({sim:.2f}) in {cat} tensor. Recommending spherical interpolation."
            }

    def get_latest_advice(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._last_result and self._last_result != self._last_applied:
                self._last_applied = self._last_result; return self._last_result
        return None
