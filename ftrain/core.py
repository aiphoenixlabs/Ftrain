import os, sys, io, time, math, json, shutil, re, random, threading
from functools import partial
import torch
from torch.utils.data import DataLoader
from unsloth import FastLanguageModel
from .config import TrainConfig
from .dataset import FtrainDataset, collate, LengthSampler
from .model_utils import seed_everything, get_family, get_num_layers, is_moe, count_params
from .speed import flash_mode
from .lora import inject as inject_lora
from .lora_dora import inject_dora
from .families import get_preset
from . import ui
from .captain import PhoenixCaptain
from .train_optim import LRFinder, adaptive_accumulation, cosine_restart_scheduler
from .data_quality import filter_by_perplexity, deduplicate, balance_datasets

class Ftrain:
    def __init__(self, config, train_data, val_data=None):
        cfg = config
        flash_mode(enabled=True, tf32=True)
        seed_everything(cfg.seed)

        if cfg.auto_resume:
            d = os.path.join(cfg.output_dir, "checkpoints")
            if os.path.isdir(d):
                steps = [f for f in os.listdir(d) if f.startswith("step_")]
                if steps:
                    steps.sort(key=lambda x: int(x.split("_")[1]))
                    cfg.resume_from_checkpoint = os.path.join(d, steps[-1])
                    print("🔄 Auto-resuming")

        family = cfg.family if cfg.family != "auto" else get_family(cfg.model_name)
        preset = get_preset(family)
        if not hasattr(cfg, 'lora_target_modules') or not cfg.lora_target_modules:
            cfg.lora_target_modules = preset["lora_targets"]

        self.config = cfg
        self.train_data = train_data
        self.val_data = val_data
        self.loss_history = []
        self.step = 0
        self.epoch = 0
        self._captain_mult = 1.0
        self._captain_layer_boosts = {"early": 1, "late": 1, "gate": 1, "router": 1, "other": 1}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cfg.use_grpo:
            try:
                from unsloth import PatchFastRL
                PatchFastRL("GRPO", FastLanguageModel)
                print("🧠 Unsloth patched for GRPO!")
            except:
                pass

        bar = ui.LoadingBar(message=f"Loading {cfg.model_name}", real_progress=cfg.show_model_progress)
        bar.start()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            kw = {"model_name": cfg.model_name, "max_seq_length": cfg.max_seq_length, "load_in_4bit": cfg.load_in_4bit}
            if not cfg.load_in_4bit:
                kw["dtype"] = torch.bfloat16
            if preset.get("attn_implementation"):
                kw["attn_implementation"] = preset["attn_implementation"]
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(**kw)
        except Exception as e:
            sys.stdout = old_stdout
            print(f"⚠️ Unsloth failed ({e}), fallback HF")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        finally:
            sys.stdout = old_stdout
            bar.done()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.captain = PhoenixCaptain(cfg) if cfg.captain_enabled else None
        if self.captain:
            self.captain.analyze_model(self.model)

        if cfg.data_perplexity_filter or cfg.data_dedup or cfg.data_sources:
            orig = len(self.train_data)
            changes = []
            if cfg.data_dedup:
                self.train_data = deduplicate(self.train_data)
                if len(self.train_data) < orig:
                    changes.append(f"Deduplication removed {orig - len(self.train_data)} duplicates.")
            if cfg.data_perplexity_filter:
                b = len(self.train_data)
                self.train_data = filter_by_perplexity(self.train_data, self.model, self.tokenizer, self.device, cfg.data_perplexity_keep_pct)
                if len(self.train_data) < b:
                    changes.append(f"Perplexity filter removed {b - len(self.train_data)} anomalies.")
            if cfg.data_sources:
                srcs = [self.train_data]
                for s in cfg.data_sources:
                    from .data_utils import load_data
                    srcs.append(load_data(s))
                self.train_data = balance_datasets(srcs, cfg.data_balance_strategy)
                changes.append("Balanced multiple data sources.")
            if self.captain:
                self.captain.analyze_and_report_data(orig, len(self.train_data), changes)

        if cfg.auto_lora_targets:
            self.model.train()
            d = self.tokenizer("Test", return_tensors="pt").to(self.device)
            out = self.model(**d, labels=d["input_ids"])
            out.loss.backward()
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = None
            gn = {}
            for n, p in self.model.named_parameters():
                if p.grad is not None and "lora" not in n:
                    m = n.split(".")[-2]
                    gn[m] = gn.get(m, 0) + p.grad.norm().item()
            cfg.lora_target_modules = [m for m, _ in sorted(gn.items(), key=lambda x: x[1], reverse=True)[:cfg.lora_target_count]]

        if cfg.use_unsloth_lora or cfg.use_dora:
            self.model = FastLanguageModel.get_peft_model(self.model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=cfg.lora_target_modules, use_dora=cfg.use_dora)
        elif cfg.use_custom_lora:
            if cfg.use_dora:
                inject_dora(self.model, cfg.lora_target_modules, cfg.lora_r, cfg.lora_alpha)
            else:
                inject_lora(self.model, cfg.lora_target_modules, cfg.lora_r, cfg.lora_alpha)

        print(f"Trainable params: {count_params(self.model)['trainable']/1e6:.2f}M")

        self.train_dataset = self.train_data if cfg.use_grpo else FtrainDataset(self.train_data, self.tokenizer, cfg.max_seq_length, cfg.use_packing)
        self.val_dataset = FtrainDataset(self.val_data, self.tokenizer, cfg.max_seq_length) if self.val_data else None
        self.total_steps = cfg.max_steps

        if cfg.use_dashboard:
            from .dashboard import TrainingDashboard
            self.dashboard = TrainingDashboard(port=cfg.dashboard_port)
            threading.Thread(target=self.dashboard.start, daemon=True).start()
        else:
            self.dashboard = None

        os.makedirs(cfg.output_dir, exist_ok=True)

    def _evaluate_model(self, p):
        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(self.model)
            inp = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inp, max_new_tokens=100, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(out[0][inp['input_ids'].shape[-1]:], skip_special_tokens=True)
        except:
            return "Error"

    def _build_opt(self):
        cfg = self.config
        am, bm = cfg.lora_a_lr_mult, cfg.lora_b_lr_mult
        e, l, g, o = [], [], [], []
        nl = get_num_layers(self.model)
        thr = max(1, nl // 3)
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            lr = cfg.learning_rate
            if "lora_A" in n:
                lr *= am
            elif "lora_B" in n:
                lr *= bm
            if "gate_proj" in n:
                g.append({"params": p, "lr": lr * cfg.swiglu_gate_boost})
            elif re.search(r"layers\.(\d+)\.", n):
                i = int(re.search(r"layers\.(\d+)\.", n).group(1))
                if i < thr:
                    e.append({"params": p, "lr": lr * cfg.layerwise_lr_decay})
                else:
                    l.append({"params": p, "lr": lr})
            else:
                o.append({"params": p, "lr": lr})
        for x in e:
            x["name"] = "early"
        for x in l:
            x["name"] = "late"
        for x in g:
            x["name"] = "gate"
        for x in o:
            x["name"] = "other"
        self.optimizer = torch.optim.AdamW(e + l + g + o, lr=cfg.learning_rate, fused=self.device.type == "cuda")
        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

    def train(self):
        cfg = self.config
        self.model.train()
        ui.fire_header()
        print(f"🧬 Model: {cfg.model_name} | Steps: {self.total_steps} | Mode: {'GRPO' if cfg.use_grpo else 'SFT'}")

        e_prompt, correct, before = "", "", ""
        if self.train_data and not cfg.use_grpo:
            s = random.choice(self.train_data)
            msgs = s.get("messages", [])
            if len(msgs) >= 2:
                e_prompt = self.tokenizer.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
                correct = msgs[1].get("content", "")
                print("\n🧠 Captain is asking the model a question before training...")
                before = self._evaluate_model(e_prompt)

        if cfg.use_grpo:
            res = self._train_grpo()
        elif cfg.use_hf_trainer:
            res = self._train_hf()
        else:
            res = self._train_custom()

        if self.captain and e_prompt:
            print("\n🧠 Captain is asking the model the same question after training...")
            after = self._evaluate_model(e_prompt)
            self.captain.evaluate_improvement(e_prompt, before, after, correct)

        return res

    def _train_grpo(self):
        cfg = self.config
        from trl import GRPOTrainer, GRPOConfig
        gd = []
        for ex in self.train_data:
            if "messages" in ex:
                pm = [m for m in ex["messages"] if m["role"] != "assistant"]
                try:
                    p = self.tokenizer.apply_chat_template(pm, tokenize=False, add_generation_prompt=True)
                except:
                    p = "\n".join(f"{m['role']}: {m['content']}" for m in pm)
                gd.append({"prompt": p, "solution": ex.get("solution", "")})
            elif "prompt" in ex:
                gd.append(ex)
        ta = GRPOConfig(
            output_dir=cfg.output_dir, max_steps=cfg.max_steps, learning_rate=cfg.learning_rate,
            logging_steps=cfg.captain_interval, save_steps=cfg.checkpoint_interval,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_generations=cfg.grpo_num_generations, max_prompt_length=512, max_completion_length=1024,
            temperature=0.7, beta=0.01, report_to="none", bf16=not cfg.load_in_4bit,
            gradient_checkpointing=cfg.gradient_checkpointing_enable, remove_unused_columns=False
        )
        tr = GRPOTrainer(model=self.model, processing_class=self.tokenizer, reward_funcs=cfg.grpo_reward_funcs, args=ta, train_dataset=gd)
        tr.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
        fp = os.path.join(cfg.output_dir, "final")
        self.model.save_pretrained(fp)
        self.tokenizer.save_pretrained(fp)
        if self.dashboard:
            self.dashboard.stop()
        ui.print_final_summary({"Model": cfg.model_name, "Steps": self.total_steps, "Mode": "GRPO", "Dir": fp})
        return self.model

    def _train_hf(self):
        cfg = self.config
        self._build_opt()
        cb = None
        if cfg.captain_enabled:
            from .callbacks import PhoenixCaptainCallback
            cb = PhoenixCaptainCallback(cfg, self.model, self.tokenizer, self.train_dataset, self.dashboard)
        try:
            if cfg.use_unsloth_trainer:
                from unsloth import UnslothTrainer, UnslothTrainingArguments as UTA
                ta = UTA(
                    output_dir=cfg.output_dir, max_steps=cfg.max_steps,
                    per_device_train_batch_size=cfg.per_device_batch_size,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    learning_rate=cfg.learning_rate, warmup_ratio=cfg.warmup_ratio,
                    logging_steps=cfg.captain_interval,
                    eval_strategy="steps" if self.val_dataset else "no",
                    eval_steps=cfg.eval_interval if self.val_dataset else 500,
                    save_strategy="steps", save_steps=cfg.checkpoint_interval,
                    save_total_limit=cfg.save_total_limit, bf16=not cfg.load_in_4bit,
                    gradient_checkpointing=cfg.gradient_checkpointing_enable,
                    dataloader_num_workers=cfg.dataloader_num_workers,
                    report_to=cfg.report_to, remove_unused_columns=False,
                    max_grad_norm=cfg.max_grad_norm, seed=cfg.seed
                )
                tr = UnslothTrainer(
                    model=self.model, tokenizer=self.tokenizer, args=ta,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.val_dataset if self.val_dataset else None,
                    data_collator=partial(collate, pad_token_id=self.tokenizer.pad_token_id or 0),
                    optimizers=(self.optimizer, None), callbacks=[cb] if cb else None
                )
            else:
                raise ImportError("Forced HF Trainer")
        except ImportError:
            from transformers import Trainer, TrainingArguments as TA
            ta = TA(
                output_dir=cfg.output_dir, max_steps=cfg.max_steps,
                per_device_train_batch_size=cfg.per_device_batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                learning_rate=cfg.learning_rate, warmup_ratio=cfg.warmup_ratio,
                logging_steps=cfg.captain_interval,
                eval_strategy="steps" if self.val_dataset else "no",
                eval_steps=cfg.eval_interval if self.val_dataset else 500,
                save_strategy="steps", save_steps=cfg.checkpoint_interval,
                save_total_limit=cfg.save_total_limit, bf16=not cfg.load_in_4bit,
                gradient_checkpointing=cfg.gradient_checkpointing_enable,
                dataloader_num_workers=cfg.dataloader_num_workers,
                report_to=cfg.report_to, remove_unused_columns=False,
                max_grad_norm=cfg.max_grad_norm
            )
            tr = Trainer(
                model=self.model, tokenizer=self.tokenizer, args=ta,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset if self.val_dataset else None,
                data_collator=partial(collate, pad_token_id=self.tokenizer.pad_token_id or 0),
                optimizers=(self.optimizer, None), callbacks=[cb] if cb else None
            )
        tr.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
        fp = os.path.join(cfg.output_dir, "final")
        self.model.save_pretrained(fp)
        self.tokenizer.save_pretrained(fp)
        if self.dashboard:
            self.dashboard.stop()
        ui.print_final_summary({"Model": cfg.model_name, "Steps": self.total_steps, "Dir": fp})
        return self.model

    def _train_custom(self):
        cfg = self.config
        self._build_opt()
        self._build_sched()
        if cfg.captain_enabled and cfg.captain_model:
            self.captain.set_family_context("generic", is_moe(self.model))
            if self.train_dataset:
                self.captain.analyze_data(self.train_dataset, self.tokenizer)
        loader = self._dataloader(self.train_dataset, True)
        it = iter(loader)
        ep = 0
        acc_loss, acc_steps = 0.0, 0
        vl = None
        ca = cfg.gradient_accumulation_steps
        msg = ""
        while self.step < self.total_steps:
            try:
                batch = next(it)
            except StopIteration:
                ep += 1
                if hasattr(loader, 'sampler'):
                    loader.sampler.set_epoch(ep)
                it = iter(loader)
                batch = next(it)
            if cfg.use_adaptive_accumulation:
                ca = adaptive_accumulation(cfg.gradient_accumulation_steps, batch["input_ids"].numel(), cfg.target_batch_tokens)
            ids = batch.get("input_ids").to(self.device)
            am = batch.get("attention_mask", torch.ones_like(ids)).to(self.device)
            lb = batch.get("labels", ids).to(self.device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if not cfg.load_in_4bit else torch.float16):
                out = self.model(input_ids=ids, attention_mask=am, labels=lb)
            loss = out.loss / ca
            loss.backward()
            acc_loss += loss.item() * ca
            acc_steps += 1
            if acc_steps >= ca:
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm).item()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.step += 1
                rl = acc_loss / acc_steps
                self.loss_history.append(rl)
                lr = self.optimizer.param_groups[0]["lr"]
                if self.captain and self.step % cfg.captain_interval == 0:
                    e, l, g = 0.0, 0.0, 0.0
                    for pg in self.optimizer.param_groups:
                        n = pg.get("name", "other")
                        grads = [p.grad for p in pg["params"] if p.grad is not None]
                        if not grads:
                            continue
                        t = sum(x.item()**2 for x in torch._foreach_norm(grads))
                        if n == "early":
                            e += t**0.5
                        elif n == "late":
                            l += t**0.5
                        elif n == "gate":
                            g += t**0.5
                    self.captain.inspect_training(self.step, rl, lr, gn, (e, l, g), vl)
                    adv = self.captain.get_latest_advice()
                    if adv:
                        msg = f"{adv['action']} (x{adv['mult']:.2f})"
                        self._captain_mult = adv["mult"]
                        lb_ = adv.get("layer_boost", "none")
                        if lb_ == "all":
                            self._captain_layer_boosts = {k: 2.0 for k in self._captain_layer_boosts}
                        else:
                            self._captain_layer_boosts = {k: 2.0 if k == lb_ else 1.0 for k in self._captain_layer_boosts}
                if self.step % cfg.eval_interval == 0 and self.val_dataset:
                    vl = self.validate()
                ui.print_train_table(self.step, self.total_steps, rl, vl, lr, gn, msg)
                if self.dashboard:
                    self.dashboard.log_metric(self.step, rl, lr, vl)
                acc_loss, acc_steps = 0.0, 0
                if self.step % cfg.checkpoint_interval == 0:
                    self.save_checkpoint(self.step)
        self.save_checkpoint(self.step, True)
        if self.dashboard:
            self.dashboard.stop()
        ui.print_final_summary({"Model": cfg.model_name, "Steps": self.step, "Loss": f"{rl:.4f}", "Dir": cfg.output_dir})
        return self.model

    def _build_sched(self):
        cfg = self.config
        tot = max(1, self.total_steps)
        warm = cfg.warmup_steps if cfg.warmup_steps > 0 else int(cfg.warmup_ratio * tot)
        if cfg.use_cosine_restarts:
            self.scheduler = cosine_restart_scheduler(self.optimizer, cfg.learning_rate, cfg.learning_rate * cfg.min_lr_ratio, warm, tot, cfg.restart_interval)
        else:
            lls = []
            for g in self.optimizer.param_groups:
                gn = g.get("name", "other")
                def mk(gn):
                    def lam(s):
                        if s < warm:
                            b = s / max(1, warm)
                        else:
                            pr = min(1.0, max(0.0, (s - warm) / max(1, tot - warm)))
                            b = cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * pr))
                        return b * self._captain_mult * self._captain_layer_boosts.get(gn, 1.0)
                    return lam
                lls.append(mk(gn))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lls)

    def _dataloader(self, ds, sh=True):
        cfg = self.config
        s = LengthSampler(ds.lengths, cfg.per_device_batch_size, sh, cfg.seed)
        return DataLoader(ds, batch_size=cfg.per_device_batch_size, sampler=s, collate_fn=partial(collate, pad_token_id=self.tokenizer.pad_token_id or 0), num_workers=cfg.dataloader_num_workers, pin_memory=cfg.pin_memory)

    def validate(self):
        if not self.val_dataset:
            return None
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        loader = self._dataloader(self.val_dataset, False)
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in loader:
                ids = b.get("input_ids").to(self.device)
                am = b.get("attention_mask", torch.ones_like(ids)).to(self.device)
                lb = b.get("labels", ids).to(self.device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if not self.config.load_in_4bit else torch.float16):
                    out = self.model(input_ids=ids, attention_mask=am, labels=lb)
                tot += out.loss.item()
                n += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.train()
        return tot / max(1, n)

    def save_checkpoint(self, step, final=False):
        cfg = self.config
        tag = "final" if final else f"step_{step}"
        path = os.path.join(cfg.output_dir, "checkpoints", tag)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if not final:
            d = os.path.join(cfg.output_dir, "checkpoints")
            if os.path.isdir(d):
                ents = [e for e in os.listdir(d) if e.startswith("step_") and "best" not in e and "final" not in e]
                ents.sort(key=lambda e: int(e.split("_")[1]))
                while len(ents) > cfg.save_total_limit:
                    shutil.rmtree(os.path.join(d, ents.pop(0)), ignore_errors=True)
        print(f"💾 checkpoint → {path}")
