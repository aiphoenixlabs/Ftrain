import queue, logging, pandas as pd
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class TrainingDashboard:
    def __init__(self, port=7860):
        self.queue = queue.Queue()
        self.port = port
        self.running = False
        self.history = {"step": [], "loss": [], "lr": [], "val_loss": []}

    def start(self):
        try:
            import gradio as gr
        except:
            logger.warning("Gradio missing.")
            return
        self.running = True
        self._launch(gr)

    def _launch(self, gr):
        def fetch():
            while not self.queue.empty():
                try:
                    i = self.queue.get_nowait()
                    self.history["step"].append(i["step"])
                    self.history["loss"].append(i["loss"])
                    self.history["lr"].append(i["lr"])
                    self.history["val_loss"].append(i["val_loss"] if i["val_loss"] else float('nan'))
                except:
                    break
            df = pd.DataFrame(self.history)
            return df if not df.empty else pd.DataFrame({"step": [0], "loss": [0], "lr": [0], "val_loss": [float('nan')]})
        with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
            gr.Markdown("# 🔥 FTRAIN Live Dashboard")
            with gr.Row():
                lp = gr.LinePlot(x="step", y="loss", title="Loss", width=400, height=300)
                vp = gr.LinePlot(x="step", y="val_loss", title="Val Loss", width=400, height=300)
            rp = gr.LinePlot(x="step", y="lr", title="LR", width=800, height=200)
            demo.load(lambda: [fetch()]*3, outputs=[lp, vp, rp], every=2)
        demo.launch(server_port=self.port, share=False, prevent_thread_lock=True, quiet=True)

    def stop(self):
        self.running = False

    def log_metric(self, step, loss, lr, val_loss=None):
        if self.running:
            self.queue.put({"step": step, "loss": loss, "lr": lr, "val_loss": val_loss})
