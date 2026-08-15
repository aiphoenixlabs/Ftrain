import torch
from transformers import TrainerCallback
from . import ui
from .captain import PhoenixCaptain
from .model_utils import is_moe

class PhoenixCaptainCallback(TrainerCallback):
    def __init__(self, cfg, model, tokenizer, train_dataset, dashboard=None):
        self.cfg = cfg
        self.model = model
        self.dashboard = dashboard
        self.captain = PhoenixCaptain(cfg)
        self.captain.set_family_context(cfg.family if cfg.family != "auto" else "generic", is_moe(model))
        self.captain.analyze_model(model)
        if train_dataset:
            self.captain.analyze_data(train_dataset, tokenizer)
        self._captain_mult = 1.0
        self._captain_layer_boosts = {"early": 1, "late": 1, "gate": 1, "router": 1, "other": 1}

    def _compute_brain_activity(self, opt):
        e, l, g = 0.0, 0.0, 0.0
        for pg in opt.param_groups:
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
        return (e, l, g)

    def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.cfg.captain_interval == 0:
            logs = state.log_history[-1] if state.log_history else {}
            loss = logs.get("loss", 0.0)
            lr = logs.get("learning_rate", self.cfg.learning_rate)
            grad_norm = logs.get("grad_norm", 0.0)
            val_loss = logs.get("eval_loss", None)

            brain = self._compute_brain_activity(optimizer)
            self.captain.inspect_training(state.global_step, loss, lr, grad_norm, brain, val_loss)
            advice = self.captain.get_latest_advice()

            if advice:
                ui.print_train_table(state.global_step, self.cfg.max_steps, loss, val_loss, lr, grad_norm, f"{advice['action']} (x{advice['mult']:.2f})")
                if self.dashboard:
                    self.dashboard.log_metric(state.global_step, loss, lr, val_loss)

                self._captain_mult = advice["mult"]
                lb = advice.get("layer_boost", "none")
                if lb == "all":
                    self._captain_layer_boosts = {k: 2.0 for k in self._captain_layer_boosts}
                else:
                    self._captain_layer_boosts = {k: 2.0 if k == lb else 1.0 for k in self._captain_layer_boosts}

                for pg in optimizer.param_groups:
                    pg["lr"] = pg.get("initial_lr", pg["lr"]) * self._captain_mult * self._captain_layer_boosts.get(pg.get("name", "other"), 1.0)
