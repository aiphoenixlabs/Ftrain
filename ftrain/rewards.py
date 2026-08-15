import re

def xml_format_reward(prompts, completions, **kwargs):
    p = r"^\s*<answer>.*?</answer>$"
    return [1.0 if re.match(p, c[0]["content"], re.DOTALL) else 0.0 for c in completions]

def math_exact_reward(prompts, completions, **kwargs):
    sols = kwargs.get("solution", [None] * len(completions))
    if not isinstance(sols, list):
        sols = [sols] * len(completions)
    scores = []
    for c, s in zip(completions, sols):
        m = re.search(r"<answer>(.*?)</answer>", c[0]["content"], re.DOTALL)
        if m and m.group(1).strip() == str(s).strip():
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores

def python_exec_reward(prompts, completions, **kwargs):
    sols = kwargs.get("solution", [None] * len(completions))
    if not isinstance(sols, list):
        sols = [sols] * len(completions)
    scores = []
    for c, s in zip(completions, sols):
        m = re.search(r"<code>(.*?)</code>", c[0]["content"], re.DOTALL)
        if not m:
            scores.append(0.0)
            continue
        try:
            import io, contextlib
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                exec(m.group(1), {"__builtins__": {}})
            out = f.getvalue().strip()
            scores.append(1.0 if out == str(s).strip() else 0.0)
        except Exception:
            scores.append(0.0)
    return scores
