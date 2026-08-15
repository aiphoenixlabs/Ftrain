import math, torch, numpy as np, logging
from typing import Iterator

logger = logging.getLogger(__name__)

class LRFinder:
    def __init__(self, model, opt, dev, start=1e-7, end=10.0, n=100):
        self.model=model
        self.opt=opt
        self.dev=dev
        self.start=start
        self.end=end
        self.n=n
        self.lrs=[]
        self.losses=[]

    def range_test(self, loader):
        self.model.train()
        mult = (self.end/self.start)**(1.0/self.n)
        lr = self.start
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        avg, best = 0.0, float('inf')
        for i, b in enumerate(loader):
            if i >= self.n:
                break
            b = {k: v.to(self.dev) for k, v in b.items()}
            l = self.model(**b).loss
            avg = 0.9*avg + 0.1*l.item() if i > 0 else l.item()
            self.lrs.append(lr)
            self.losses.append(avg)
            l.backward()
            self.opt.step()
            self.opt.zero_grad()
            lr *= mult
            for pg in self.opt.param_groups:
                pg['lr'] = lr
            if avg < best:
                best = avg
            if avg > 4*best or torch.isnan(l):
                break
        return self.suggest()

    def suggest(self):
        if not self.losses:
            return 1e-4
        ls = np.array(self.losses)
        lrs = np.array(self.lrs)
        sl = np.convolve(ls, np.ones(5)/5, mode='valid')
        if len(sl) < 2:
            return lrs[np.argmin(ls)]
        al = lrs[(len(lrs)-len(sl))//2:(len(lrs)+len(sl))//2]
        return float(al[np.argmin(np.gradient(sl)/np.gradient(al))]/10.0)

def adaptive_accumulation(ca, bt, t=8192):
    return min(max(1, int(t/bt)), ca*2) if bt > 0 else ca

def cosine_restart_scheduler(opt, base, mn, warm, tot, ri):
    def lam(s):
        if s < warm:
            return s / max(1, warm)
        c = s // ri
        si = s % ri
        ci = math.pi * si / ri
        return (mn + 0.5 * (base - mn) * (1 + math.cos(ci)) / (c + 1)) / base if base > 0 else 0.0
    return torch.optim.lr_scheduler.LambdaLR(opt, lam)
