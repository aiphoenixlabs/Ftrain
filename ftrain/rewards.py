"""
FTRAIN Reward Functions v1.1
============================

Robust reward functions for FTRAIN / TRL-style GRPO training.

Public rewards
--------------
xml_format_reward
    Rewards responses that contain exactly one valid <answer>...</answer>
    block with no meaningful text outside the block.

math_exact_reward
    Rewards mathematically correct final answers while tolerating harmless
    formatting differences such as commas, whitespace, $ signs, percentages,
    boxed answers and equivalent numeric representations.

python_exec_reward
    Rewards generated Python snippets whose output matches the expected
    solution.

Design goals
------------
• Compatible with common TRL completion formats.
• Defensive handling of malformed completions.
• No reward-function crashes caused by bad model output.
• Batch-length consistency.
• Efficient regex compilation.
• Better exact/numeric math matching.
• Optional strict Python execution through AST validation.
• No unrestricted ``exec``.
• Clear, deterministic reward behavior.
• Easy extension with additional reward functions.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import operator
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# =============================================================================
# Constants
# =============================================================================

REWARD_OK = 1.0
REWARD_FAIL = 0.0

_XML_ANSWER_RE = re.compile(
    r"^\s*<answer>\s*(.*?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)

_XML_ANY_ANSWER_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL | re.IGNORECASE,
)

_CODE_RE = re.compile(
    r"<code>\s*(.*?)\s*</code>",
    re.DOTALL | re.IGNORECASE,
)

_THINK_RE = re.compile(
    r"<think>\s*(.*?)\s*</think>",
    re.DOTALL | re.IGNORECASE,
)

_BOXED_RE = re.compile(
    r"\\boxed\s*\{\s*(.*?)\s*\}",
    re.DOTALL,
)

_FINAL_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])
    [-+]?
    (?:
        (?:\d{1,3}(?:,\d{3})+)
        |
        (?:\d+)
    )
    (?:\.\d+)?
    (?:[eE][-+]?\d+)?
    %?
    (?![\w.])
    """,
    re.VERBOSE,
)


# =============================================================================
# Generic completion extraction
# =============================================================================

def _completion_to_text(
    completion: Any,
) -> str:
    """
    Convert common TRL/Transformers completion formats into plain text.

    Supported examples
    -------------------
    "hello"

    [{"role": "assistant", "content": "hello"}]

    {"content": "hello"}

    {"text": "hello"}

    [{"content": "..."}]
    """
    if completion is None:
        return ""

    if isinstance(completion, str):
        return completion

    if isinstance(completion, Mapping):
        if "content" in completion:
            return str(
                completion.get("content") or ""
            )

        if "text" in completion:
            return str(
                completion.get("text") or ""
            )

        # Sometimes chat messages are nested.
        if "messages" in completion:
            return _messages_to_text(
                completion.get("messages")
            )

        return ""

    if isinstance(completion, Sequence):
        # TRL commonly returns a list of role/content dictionaries.
        if not isinstance(
            completion,
            (str, bytes),
        ):
            parts: List[str] = []

            for item in completion:
                text = _completion_to_text(
                    item
                )

                if text:
                    parts.append(text)

            return "\n".join(parts)

    return str(completion)


def _messages_to_text(
    messages: Any,
) -> str:
    if not isinstance(
        messages,
        Sequence,
    ) or isinstance(
        messages,
        (str, bytes),
    ):
        return ""

    parts: List[str] = []

    for message in messages:
        if not isinstance(
            message,
            Mapping,
        ):
            continue

        content = message.get(
            "content",
            "",
        )

        if content is None:
            continue

        parts.append(
            str(content)
        )

    return "\n".join(parts)


def _completion_texts(
    completions: Any,
) -> List[str]:
    """
    Normalize an entire reward batch into strings.
    """
    if completions is None:
        return []

    if isinstance(
        completions,
        str,
    ):
        return [completions]

    if isinstance(
        completions,
        Mapping,
    ):
        return [
            _completion_to_text(
                completions
            )
        ]

    if isinstance(
        completions,
        Sequence,
    ) and not isinstance(
        completions,
        (str, bytes),
    ):
        return [
            _completion_to_text(
                completion
            )
            for completion in completions
        ]

    return [
        _completion_to_text(
            completions
        )
    ]


# =============================================================================
# Batch helpers
# =============================================================================

def _batch_values(
    value: Any,
    size: int,
) -> List[Any]:
    """
    Normalize a scalar/list/tuple batch argument to exactly ``size`` items.
    """
    if size <= 0:
        return []

    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (str, bytes),
    ):
        values = list(value)

        if len(values) >= size:
            return values[:size]

        if not values:
            return [None] * size

        # Repeat the last value instead of crashing on a shorter batch.
        return (
            values
            + [values[-1]]
            * (size - len(values))
        )

    return [value] * size


def _solutions_from_kwargs(
    kwargs: Mapping[str, Any],
    size: int,
) -> List[Any]:
    """
    Extract expected answers from common field names.
    """
    for key in (
        "solution",
        "solutions",
        "answer",
        "answers",
        "target",
        "targets",
        "ground_truth",
        "ground_truths",
    ):
        if key in kwargs:
            return _batch_values(
                kwargs.get(key),
                size,
            )

    return [None] * size


# =============================================================================
# XML / answer extraction
# =============================================================================

def _extract_answer(
    text: str,
) -> Optional[str]:
    match = _XML_ANY_ANSWER_RE.search(
        text
    )

    if match is None:
        return None

    return match.group(1).strip()


def _has_strict_answer_format(
    text: str,
) -> bool:
    return bool(
        _XML_ANSWER_RE.fullmatch(
            text
        )
    )


def _strip_reasoning(
    text: str,
) -> str:
    """
    Remove optional <think>...</think> blocks when comparing final answers.
    """
    return _THINK_RE.sub(
        " ",
        text,
    ).strip()


# =============================================================================
# Normalization
# =============================================================================

def _normalize_text(
    value: Any,
) -> str:
    """
    Normalize harmless textual formatting without changing semantic content.
    """
    if value is None:
        return ""

    text = str(value)
    text = text.strip()

    text = text.replace(
        "\u00a0",
        " ",
    )

    # Normalize common answer wrappers.
    boxed = _BOXED_RE.search(
        text
    )

    if boxed is not None:
        text = boxed.group(1).strip()

    text = text.strip(
        " \t\r\n`'\""
    )

    # Remove harmless LaTeX-style wrappers around simple answers.
    text = re.sub(
        r"^\$+|\$+$",
        "",
        text,
    )

    text = re.sub(
        r"\\(?:text|mathrm|operatorname)\s*\{\s*(.*?)\s*\}",
        r"\1",
        text,
        flags=re.DOTALL,
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _numeric_value(
    value: Any,
) -> Optional[float]:
    """
    Parse a simple numeric answer.

    Handles:
        123
        123.5
        1,234.5
        $123.50
        25%
        -3e-2
        \\boxed{42}
    """
    if value is None:
        return None

    text = _normalize_text(
        value
    )

    if not text:
        return None

    # Percentage.
    percent = text.endswith(
        "%"
    )

    if percent:
        text = text[:-1].strip()

    # Currency / common symbols.
    text = text.replace(
        "$",
        "",
    )

    text = text.replace(
        "€",
        "",
    )

    text = text.replace(
        "£",
        "",
    )

    text = text.replace(
        ",",
        "",
    )

    # Simple direct conversion first.
    try:
        number = float(text)

        if percent:
            number /= 100.0

        if math.isfinite(number):
            return number

    except (
        TypeError,
        ValueError,
    ):
        pass

    # Try a numeric token inside a sentence.
    match = _FINAL_NUMBER_RE.search(
        text
    )

    if match is None:
        return None

    token = match.group(0)

    if token.endswith("%"):
        token = token[:-1]

        try:
            number = float(
                token.replace(",", "")
            ) / 100.0

            return (
                number
                if math.isfinite(number)
                else None
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    try:
        number = float(
            token.replace(",", "")
        )

        return (
            number
            if math.isfinite(number)
            else None
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def _math_equivalent(
    predicted: Any,
    expected: Any,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> bool:
    """
    Compare mathematical answers using exact normalized text first and then
    numeric equivalence.
    """
    predicted_text = _normalize_text(
        predicted
    )

    expected_text = _normalize_text(
        expected
    )

    if not predicted_text:
        return False

    if not expected_text:
        return False

    # Exact normalized text.
    if predicted_text.casefold() == expected_text.casefold():
        return True

    predicted_number = _numeric_value(
        predicted_text
    )

    expected_number = _numeric_value(
        expected_text
    )

    if (
        predicted_number is None
        or expected_number is None
    ):
        return False

    tolerance = max(
        atol,
        rtol
        * max(
            abs(expected_number),
            1.0,
        ),
    )

    return abs(
        predicted_number
        - expected_number
    ) <= tolerance


# =============================================================================
# XML format reward
# =============================================================================

def xml_format_reward(
    prompts: Any,
    completions: Any,
    **kwargs: Any,
) -> List[float]:
    """
    Reward strict:

        <answer>...</answer>

    formatting.

    The complete response must consist only of the answer tag, apart from
    surrounding whitespace.
    """
    del prompts, kwargs

    texts = _completion_texts(
        completions
    )

    return [
        REWARD_OK
        if _has_strict_answer_format(text)
        else REWARD_FAIL
        for text in texts
    ]


# =============================================================================
# Math exact reward
# =============================================================================

def math_exact_reward(
    prompts: Any,
    completions: Any,
    **kwargs: Any,
) -> List[float]:
    """
    Reward mathematically correct <answer>...</answer> outputs.

    The reward is 1.0 when:
        - a valid answer block exists, and
        - its normalized value equals the expected solution.

    Numeric answers use tolerance-aware comparison so that:
        18
        18.0
        18.000
        18,00?  # normal parser rules apply
    can be treated consistently.
    """
    del prompts

    texts = _completion_texts(
        completions
    )

    solutions = _solutions_from_kwargs(
        kwargs,
        len(texts),
    )

    scores: List[float] = []

    for text, expected in zip(
        texts,
        solutions,
    ):
        answer = _extract_answer(
            text
        )

        if answer is None:
            # As a convenience, if the model did not follow XML but the
            # completion contains a standalone final answer, don't give full
            # reward. We still require the answer tag for this reward.
            scores.append(
                REWARD_FAIL
            )
            continue

        if _math_equivalent(
            answer,
            expected,
        ):
            scores.append(
                REWARD_OK
            )
        else:
            scores.append(
                REWARD_FAIL
            )

    return scores


# =============================================================================
# Safer Python execution
# =============================================================================

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_ALLOWED_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
}

_ALLOWED_MATH = {
    name: getattr(
        math,
        name,
    )
    for name in (
        "ceil",
        "floor",
        "sqrt",
        "fabs",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "exp",
        "log",
        "log10",
        "pow",
        "factorial",
    )
    if hasattr(
        math,
        name,
    )
}


class _SafePythonEvaluator:
    """
    Small AST interpreter for deterministic arithmetic/programming rewards.

    This intentionally does NOT execute arbitrary Python bytecode.

    Supported:
        numbers
        strings
        lists/tuples
        arithmetic
        comparisons
        boolean operations
        assignments to local variables
        if statements
        for loops over small ranges/lists
        print()
        whitelisted math/builtin functions

    Unsupported:
        imports
        attribute traversal
        file I/O
        subprocess
        classes
        lambdas
        arbitrary function definitions
        comprehensions
        async constructs
        dunder access
    """

    MAX_STEPS = 20_000
    MAX_RANGE = 10_000
    MAX_OUTPUT = 16_384
    MAX_POWER = 100_000

    def __init__(self) -> None:
        self.env: Dict[str, Any] = {}
        self.output = io.StringIO()
        self.steps = 0

        self.functions = {
            **_ALLOWED_BUILTINS,
            **_ALLOWED_MATH,
        }

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------

    def run(
        self,
        source: str,
    ) -> str:
        tree = ast.parse(
            source,
            mode="exec",
        )

        self._validate_tree(
            tree
        )

        self._exec_block(
            tree.body
        )

        output = self.output.getvalue()

        if len(output) > self.MAX_OUTPUT:
            output = output[: self.MAX_OUTPUT]

        return output.strip()

    # -------------------------------------------------------------------------
    # Safety validation
    # -------------------------------------------------------------------------

    def _tick(self) -> None:
        self.steps += 1

        if self.steps > self.MAX_STEPS:
            raise RuntimeError(
                "Python reward execution exceeded the step limit."
            )

    def _validate_tree(
        self,
        tree: ast.AST,
    ) -> None:
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Lambda,
                    ast.With,
                    ast.AsyncWith,
                    ast.Try,
                    ast.Raise,
                    ast.Delete,
                    ast.Global,
                    ast.Nonlocal,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.Await,
                    ast.AsyncFor,
                    ast.AsyncFunctionDef,
                    ast.NamedExpr,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                    ast.Match,
                ),
            ):
                raise ValueError(
                    f"Unsupported Python construct: "
                    f"{type(node).__name__}"
                )

            if isinstance(
                node,
                ast.Attribute,
            ):
                # No x.y access: this eliminates the most common escape route.
                raise ValueError(
                    "Attribute access is disabled."
                )

            if isinstance(
                node,
                ast.Name,
            ):
                if (
                    node.id.startswith("_")
                    and node.id not in {"_"}
                ):
                    raise ValueError(
                        "Private names are disabled."
                    )

            if isinstance(
                node,
                ast.Constant,
            ):
                if isinstance(
                    node.value,
                    (bytes, bytearray),
                ):
                    raise ValueError(
                        "Byte constants are disabled."
                    )

    # -------------------------------------------------------------------------
    # Statement evaluator
    # -------------------------------------------------------------------------

    def _exec_block(
        self,
        statements: Sequence[ast.stmt],
    ) -> None:
        for statement in statements:
            self._tick()
            self._exec_statement(
                statement
            )

    def _exec_statement(
        self,
        node: ast.stmt,
    ) -> None:
        if isinstance(
            node,
            ast.Expr,
        ):
            self._eval_expr(
                node.value
            )
            return

        if isinstance(
            node,
            ast.Assign,
        ):
            value = self._eval_expr(
                node.value
            )

            for target in node.targets:
                self._assign(
                    target,
                    value,
                )

            return

        if isinstance(
            node,
            ast.AnnAssign,
        ):
            value = (
                self._eval_expr(
                    node.value
                )
                if node.value is not None
                else None
            )

            self._assign(
                node.target,
                value,
            )
            return

        if isinstance(
            node,
            ast.AugAssign,
        ):
            current = self._eval_expr(
                node.target
            )

            value = self._eval_expr(
                node.value
            )

            operation = _ALLOWED_BINOPS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Unsupported augmented operator."
                )

            self._assign(
                node.target,
                operation(
                    current,
                    value,
                ),
            )
            return

        if isinstance(
            node,
            ast.If,
        ):
            condition = self._eval_expr(
                node.test
            )

            if condition:
                self._exec_block(
                    node.body
                )
            else:
                self._exec_block(
                    node.orelse
                )

            return

        if isinstance(
            node,
            ast.For,
        ):
            iterable = self._eval_expr(
                node.iter
            )

            values = list(
                iterable
            )

            if len(values) > self.MAX_RANGE:
                raise ValueError(
                    "Loop is too large."
                )

            for value in values:
                self._tick()

                self._assign(
                    node.target,
                    value,
                )

                self._exec_block(
                    node.body
                )

            self._exec_block(
                node.orelse
            )

            return

        if isinstance(
            node,
            ast.While,
        ):
            iterations = 0

            while self._eval_expr(
                node.test
            ):
                iterations += 1

                if iterations > self.MAX_RANGE:
                    raise ValueError(
                        "While loop exceeded the safety limit."
                    )

                self._tick()

                self._exec_block(
                    node.body
                )

            self._exec_block(
                node.orelse
            )

            return

        if isinstance(
            node,
            ast.Pass,
        ):
            return

        raise ValueError(
            f"Unsupported statement: "
            f"{type(node).__name__}"
        )

    # -------------------------------------------------------------------------
    # Expression evaluator
    # -------------------------------------------------------------------------

    def _eval_expr(
        self,
        node: ast.AST,
    ) -> Any:
        self._tick()

        if isinstance(
            node,
            ast.Constant,
        ):
            return node.value

        if isinstance(
            node,
            ast.Name,
        ):
            if node.id in self.env:
                return self.env[
                    node.id
                ]

            if node.id in self.functions:
                return self.functions[
                    node.id
                ]

            raise NameError(
                f"Unknown name: {node.id}"
            )

        if isinstance(
            node,
            ast.List,
        ):
            return [
                self._eval_expr(
                    element
                )
                for element in node.elts
            ]

        if isinstance(
            node,
            ast.Tuple,
        ):
            return tuple(
                self._eval_expr(
                    element
                )
                for element in node.elts
            )

        if isinstance(
            node,
            ast.Set,
        ):
            return {
                self._eval_expr(
                    element
                )
                for element in node.elts
            }

        if isinstance(
            node,
            ast.Dict,
        ):
            return {
                self._eval_expr(key): self._eval_expr(value)
                for key, value in zip(
                    node.keys,
                    node.values,
                )
            }

        if isinstance(
            node,
            ast.BinOp,
        ):
            operation = _ALLOWED_BINOPS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Unsupported binary operator."
                )

            left = self._eval_expr(
                node.left
            )
            right = self._eval_expr(
                node.right
            )

            # Prevent giant exponentiation.
            if (
                isinstance(
                    node.op,
                    ast.Pow,
                )
                and abs(
                    float(right)
                ) > 20
            ):
                raise ValueError(
                    "Exponent is too large."
                )

            result = operation(
                left,
                right,
            )

            if isinstance(
                result,
                (int, float),
            ):
                if abs(
                    float(result)
                ) > self.MAX_POWER:
                    raise ValueError(
                        "Numeric result is too large."
                    )

            return result

        if isinstance(
            node,
            ast.UnaryOp,
        ):
            operation = _ALLOWED_UNARYOPS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Unsupported unary operator."
                )

            return operation(
                self._eval_expr(
                    node.operand
                )
            )

        if isinstance(
            node,
            ast.BoolOp,
        ):
            if isinstance(
                node.op,
                ast.And,
            ):
                result = True

                for value in node.values:
                    result = self._eval_expr(
                        value
                    )

                    if not result:
                        return result

                return result

            if isinstance(
                node.op,
                ast.Or,
            ):
                result = False

                for value in node.values:
                    result = self._eval_expr(
                        value
                    )

                    if result:
                        return result

                return result

            raise ValueError(
                "Unsupported boolean operator."
            )

        if isinstance(
            node,
            ast.Compare,
        ):
            left = self._eval_expr(
                node.left
            )

            for operation_node, comparator_node in zip(
                node.ops,
                node.comparators,
            ):
                operation = _ALLOWED_CMPOPS.get(
                    type(operation_node)
                )

                if operation is None:
                    raise ValueError(
                        "Unsupported comparison operator."
                    )

                right = self._eval_expr(
                    comparator_node
                )

                if not operation(
                    left,
                    right,
                ):
                    return False

                left = right

            return True

        if isinstance(
            node,
            ast.IfExp,
        ):
            return (
                self._eval_expr(
                    node.body
                )
                if self._eval_expr(
                    node.test
                )
                else self._eval_expr(
                    node.orelse
                )
            )

        if isinstance(
            node,
            ast.Subscript,
        ):
            value = self._eval_expr(
                node.value
            )

            index = self._eval_expr(
                node.slice
            )

            return value[
                index
            ]

        if isinstance(
            node,
            ast.Slice,
        ):
            lower = (
                self._eval_expr(
                    node.lower
                )
                if node.lower
                else None
            )

            upper = (
                self._eval_expr(
                    node.upper
                )
                if node.upper
                else None
            )

            step = (
                self._eval_expr(
                    node.step
                )
                if node.step
                else None
            )

            return slice(
                lower,
                upper,
                step,
            )

        if isinstance(
            node,
            ast.Call,
        ):
            function = self._eval_expr(
                node.func
            )

            if function not in self.functions.values():
                raise ValueError(
                    "Function call is not allowed."
                )

            args = [
                self._eval_expr(
                    argument
                )
                for argument in node.args
            ]

            kwargs = {
                keyword.arg: self._eval_expr(
                    keyword.value
                )
                for keyword in node.keywords
                if keyword.arg is not None
            }

            if function is print:
                text = " ".join(
                    str(value)
                    for value in args
                )

                self.output.write(
                    text
                )

                self.output.write(
                    "\n"
                )

                return None

            return function(
                *args,
                **kwargs,
            )

        raise ValueError(
            f"Unsupported expression: "
            f"{type(node).__name__}"
        )

    # -------------------------------------------------------------------------
    # Assignment
    # -------------------------------------------------------------------------

    def _assign(
        self,
        target: ast.AST,
        value: Any,
    ) -> None:
        if isinstance(
            target,
            ast.Name,
        ):
            if target.id.startswith("_"):
                raise ValueError(
                    "Private variable names are disabled."
                )

            self.env[
                target.id
            ] = value
            return

        if isinstance(
            target,
            (ast.Tuple, ast.List),
        ):
            if not isinstance(
                value,
                (tuple, list),
            ):
                raise ValueError(
                    "Cannot unpack non-sequence value."
                )

            if len(
                target.elts
            ) != len(value):
                raise ValueError(
                    "Unpacking count mismatch."
                )

            for child, child_value in zip(
                target.elts,
                value,
            ):
                self._assign(
                    child,
                    child_value,
                )

            return

        if isinstance(
            target,
            ast.Subscript,
        ):
            container = self._eval_expr(
                target.value
            )

            index = self._eval_expr(
                target.slice
            )

            container[
                index
            ] = value

            return

        raise ValueError(
            "Unsupported assignment target."
        )


# =============================================================================
# Python code extraction/execution
# =============================================================================

def _extract_code(
    text: str,
) -> Optional[str]:
    match = _CODE_RE.search(
        text
    )

    if match is None:
        return None

    return match.group(1).strip()


def _execute_python_safely(
    source: str,
) -> Tuple[bool, str]:
    """
    Execute generated Python using the restricted AST interpreter.

    Returns:
        (success, stdout)
    """
    try:
        evaluator = _SafePythonEvaluator()

        output = evaluator.run(
            source
        )

        return True, output

    except Exception:
        return False, ""


def _normalize_program_output(
    output: Any,
) -> str:
    text = _normalize_text(
        output
    )

    if not text:
        return ""

    # Compare the last non-empty line too, since many generated programs
    # print explanatory intermediate output before the final answer.
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        return lines[-1]

    return text


def python_exec_reward(
    prompts: Any,
    completions: Any,
    **kwargs: Any,
) -> List[float]:
    """
    Reward Python-generated answers.

    Expected generated format:

        <code>
        print(2 + 2)
        </code>

    The code is evaluated by a restricted AST interpreter rather than
    unrestricted ``exec``.

    The final printed line is compared against the expected solution using
    both normalized text and numeric equivalence.
    """
    del prompts

    texts = _completion_texts(
        completions
    )

    solutions = _solutions_from_kwargs(
        kwargs,
        len(texts),
    )

    scores: List[float] = []

    for text, expected in zip(
        texts,
        solutions,
    ):
        source = _extract_code(
            text
        )

        if not source:
            scores.append(
                REWARD_FAIL
            )
            continue

        success, output = (
            _execute_python_safely(
                source
            )
        )

        if not success:
            scores.append(
                REWARD_FAIL
            )
            continue

        predicted = _normalize_program_output(
            output
        )

        if not predicted:
            scores.append(
                REWARD_FAIL
            )
            continue

        if _math_equivalent(
            predicted,
            expected,
        ):
            scores.append(
                REWARD_OK
            )
        else:
            scores.append(
                REWARD_FAIL
            )

    return scores


# =============================================================================
# Additional useful rewards
# =============================================================================

def answer_presence_reward(
    prompts: Any,
    completions: Any,
    **kwargs: Any,
) -> List[float]:
    """
    Small shaping reward for producing an answer block at all.

    This is useful as a secondary GRPO reward.
    """
    del prompts, kwargs

    texts = _completion_texts(
        completions
    )

    return [
        REWARD_OK
        if _extract_answer(text)
        else REWARD_FAIL
        for text in texts
    ]


def answer_length_reward(
    prompts: Any,
    completions: Any,
    **kwargs: Any,
) -> List[float]:
    """
    Soft reward for concise final answers.

    This intentionally avoids rewarding extremely long answers. The reward
    increases from 0 to 1 for answers up to ``max_chars``.
    """
    del prompts

    max_chars = int(
        kwargs.get(
            "max_answer_chars",
            512,
        )
    )

    max_chars = max(
        1,
        max_chars,
    )

    texts = _completion_texts(
        completions
    )

    scores: List[float] = []

    for text in texts:
        answer = _extract_answer(
            text
        )

        if answer is None:
            scores.append(
                REWARD_FAIL
            )
            continue

        length = len(
            answer
        )

        if length <= max_chars:
            score = 1.0
        else:
            # Gracefully decay instead of dropping straight to zero.
            score = max(
                0.0,
                max_chars / max(
                    length,
                    1,
                ),
            )

        scores.append(
            float(score)
        )

    return scores


# =============================================================================
# Composite reward
# =============================================================================

def combined_math_reward(
    prompts: Any,
    completions: Any,
    **kwargs: Any,
) -> List[float]:
    """
    Balanced composite reward for mathematical GRPO training.

    Components
    ----------
    Format:
        valid <answer>...</answer>

    Correctness:
        exact/numeric solution matching

    Presence:
        answer block exists

    The final score is intentionally conservative:
        25% format
        65% correctness
        10% presence

    This can be overridden with keyword weights.
    """
    format_weight = float(
        kwargs.get(
            "format_weight",
            0.25,
        )
    )

    correctness_weight = float(
        kwargs.get(
            "correctness_weight",
            0.65,
        )
    )

    presence_weight = float(
        kwargs.get(
            "presence_weight",
            0.10,
        )
    )

    total_weight = (
        format_weight
        + correctness_weight
        + presence_weight
    )

    if total_weight <= 0:
        format_weight = 0.25
        correctness_weight = 0.65
        presence_weight = 0.10
        total_weight = 1.0

    format_weight /= total_weight
    correctness_weight /= total_weight
    presence_weight /= total_weight

    format_scores = xml_format_reward(
        prompts,
        completions,
        **kwargs,
    )

    correctness_scores = math_exact_reward(
        prompts,
        completions,
        **kwargs,
    )

    presence_scores = answer_presence_reward(
        prompts,
        completions,
        **kwargs,
    )

    return [
        (
            format_weight * format_score
            + correctness_weight * correctness_score
            + presence_weight * presence_score
        )
        for format_score, correctness_score, presence_score
        in zip(
            format_scores,
            correctness_scores,
            presence_scores,
        )
    ]


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "xml_format_reward",
    "math_exact_reward",
    "python_exec_reward",
    "answer_presence_reward",
    "answer_length_reward",
    "combined_math_reward",
    ]
