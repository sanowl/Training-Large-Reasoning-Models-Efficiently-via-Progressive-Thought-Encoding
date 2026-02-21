from __future__ import annotations

import ast
import math
import re
from typing import Sequence

import torch


_TEXT_PATTERN = re.compile(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}")
_SQRT_PATTERN = re.compile(r"\\sqrt\{([^{}]+)\}")
_FRAC_PATTERN = re.compile(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}")
_ANSWER_PREFIX_PATTERN = re.compile(r"^(?:final\s+answer|answer|ans)\s*[:=]\s*", re.IGNORECASE)
_THOUSANDS_PATTERN = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _extract_braced_content(text: str, open_idx: int) -> tuple[str, int] | None:
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "{":
        return None
    depth = 0
    for idx in range(open_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : idx], idx + 1
    return None


def _extract_boxed(text: str) -> str | None:
    best = None
    cursor = 0
    marker = "\\boxed"
    while True:
        pos = text.find(marker, cursor)
        if pos < 0:
            break
        open_idx = pos + len(marker)
        parsed = _extract_braced_content(text, open_idx)
        if parsed is None:
            cursor = open_idx
            continue
        value, next_idx = parsed
        if value.strip():
            best = value.strip()
        cursor = next_idx
    return best


def _strip_latex(text: str) -> str:
    text = text.strip()
    text = text.replace("$", "")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("\\!", "")
    text = text.replace("\\,", "")
    text = text.replace("\\;", "")
    text = text.replace("\\:", "")
    text = text.replace("\\cdot", "*")
    text = text.replace("\\times", "*")
    text = text.replace("\\div", "/")
    text = text.replace("−", "-")
    text = text.replace("–", "-")
    text = text.replace("\\pi", "pi")
    prev = None
    current = text
    while prev != current:
        prev = current
        current = _TEXT_PATTERN.sub(r"\1", current)
        current = _SQRT_PATTERN.sub(r"sqrt(\1)", current)
        current = _FRAC_PATTERN.sub(r"(\1)/(\2)", current)
    return current


def _pick_equation_side(text: str) -> str:
    if text.count("=") != 1:
        return text
    left, right = text.split("=", maxsplit=1)
    left = left.strip()
    right = right.strip()
    left_has_alpha = bool(re.search(r"[a-zA-Z]", left))
    right_has_alpha = bool(re.search(r"[a-zA-Z]", right))
    if left_has_alpha and not right_has_alpha:
        return right
    if right_has_alpha and not left_has_alpha:
        return left
    return text


def _strip_outer_parens(text: str) -> str:
    out = text
    while out.startswith("(") and out.endswith(")"):
        depth = 0
        balanced = True
        for idx, char in enumerate(out):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
                if depth == 0 and idx != len(out) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        out = out[1:-1].strip()
    return out


def _canonical_symbolic(text: str) -> str:
    text = _strip_latex(text).lower().strip()
    text = _ANSWER_PREFIX_PATTERN.sub("", text)
    text = _pick_equation_side(text)
    text = _THOUSANDS_PATTERN.sub("", text)
    text = text.replace(" ", "")
    text = text.rstrip(".,;:")
    text = _strip_outer_parens(text)
    return text


def _is_percent(text: str) -> bool:
    return text.endswith("%")


def _safe_numeric_eval(expr: str) -> float | None:
    allowed_binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
    }
    allowed_unary = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("unsupported constant")
        if isinstance(node, ast.UnaryOp):
            op_fn = allowed_unary.get(type(node.op))
            if op_fn is None:
                raise ValueError("unsupported unary op")
            return float(op_fn(evaluate(node.operand)))
        if isinstance(node, ast.BinOp):
            op_fn = allowed_binary.get(type(node.op))
            if op_fn is None:
                raise ValueError("unsupported binary op")
            return float(op_fn(evaluate(node.left), evaluate(node.right)))
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return math.pi
            if node.id == "e":
                return math.e
            raise ValueError("unsupported identifier")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id != "sqrt":
                raise ValueError("unsupported function")
            if len(node.args) != 1:
                raise ValueError("sqrt expects one arg")
            return math.sqrt(evaluate(node.args[0]))
        raise ValueError("unsupported expression")

    expr = expr.strip().replace("^", "**").replace("%", "")
    expr = re.sub(r"(\d)(pi|sqrt\()", r"\1*\2", expr)
    expr = re.sub(r"(\))(\d|pi|sqrt\()", r"\1*\2", expr)
    if not expr:
        return None
    try:
        parsed = ast.parse(expr, mode="eval")
        out = float(evaluate(parsed))
    except Exception:
        return None

    if not math.isfinite(out):
        return None
    return out


def _to_numeric(text: str) -> float | None:
    canonical = _canonical_symbolic(text)
    if not canonical:
        return None
    percent = _is_percent(canonical)
    value = _safe_numeric_eval(canonical)
    if value is None:
        return None
    if percent:
        return value / 100.0
    return value


def extract_final_answer(text: str) -> str:
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    answer = lines[-1]
    answer = _ANSWER_PREFIX_PATTERN.sub("", answer)
    return answer.strip()


def answers_equivalent(prediction: str, reference: str, *, rtol: float = 1e-4, atol: float = 1e-8) -> bool:
    pred_final = extract_final_answer(prediction)
    ref_final = extract_final_answer(reference)

    pred_num = _to_numeric(pred_final)
    ref_num = _to_numeric(ref_final)
    if pred_num is not None and ref_num is not None:
        return math.isclose(pred_num, ref_num, rel_tol=rtol, abs_tol=atol)

    pred_sym = _canonical_symbolic(pred_final)
    ref_sym = _canonical_symbolic(ref_final)
    return bool(pred_sym) and pred_sym == ref_sym


def exact_match_reward(predictions: Sequence[str], references: Sequence[str]) -> torch.Tensor:
    if len(predictions) != len(references):
        raise ValueError(f"length mismatch: {len(predictions)} predictions vs {len(references)} references")
    rewards = []
    for pred, ref in zip(predictions, references):
        pred_ans = _canonical_symbolic(extract_final_answer(pred))
        ref_ans = _canonical_symbolic(extract_final_answer(ref))
        rewards.append(1.0 if pred_ans and pred_ans == ref_ans else 0.0)
    return torch.tensor(rewards, dtype=torch.float32)


def math_equivalence_reward(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-8,
) -> torch.Tensor:
    if len(predictions) != len(references):
        raise ValueError(f"length mismatch: {len(predictions)} predictions vs {len(references)} references")
    rewards = [
        1.0 if answers_equivalent(pred, ref, rtol=rtol, atol=atol) else 0.0
        for pred, ref in zip(predictions, references)
    ]
    return torch.tensor(rewards, dtype=torch.float32)
