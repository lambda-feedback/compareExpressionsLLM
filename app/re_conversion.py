import re
from collections import Counter

def convert_diff_re(expr: str,params) -> str:
    def power_suffix(var_powers):
        suffix = []
        for var, power in var_powers:
            suffix.append(f"{var}{power}" if int(power) > 1 else var)
        return "_".join(suffix)

    # 1. diff(f, x[, x, ...]) → d_f_x, d_f_x2, d_f_x_y
    def replace_diff(match):
        var = match.group(1)
        args = match.group(2)
        vars_ = [v.strip() for v in args.split(',')]
        if not all(re.fullmatch(r"[a-zA-Z_]\w*", v) for v in vars_):
            return match.group(0)
        counter = Counter(vars_)
        parts = [(v, counter[v]) for v in dict.fromkeys(vars_)]
        return f'd_{var}_' + power_suffix(parts)

    expr = re.sub(
        r'\bdiff\(\s*([a-zA-Z_]\w*)\s*,\s*((?:[a-zA-Z_]\w*\s*,\s*)*[a-zA-Z_]\w*)\s*\)',
        replace_diff,
        expr
    )

    # 2. d**n f / dx [**n] [dy [**m]] → d_f_x2_y
    expr = re.sub(
        r'd\*\*(\d+)\s*([a-zA-Z_]\w*)\s*/\s*(d[a-zA-Z_]\w*(?:\*\*\d+)?(?:\s*d[a-zA-Z_]\w*(?:\*\*\d+)?)*)',
        lambda m: (
            f'd_{m.group(2)}_' + power_suffix([
                (v.group(1), v.group(2) or "1")
                for v in re.finditer(r'd([a-zA-Z_]\w*)(?:\*\*(\d+))?', m.group(3))
            ])
            if int(m.group(1)) == sum(
                int(v.group(2) or "1") for v in re.finditer(r'd([a-zA-Z_]\w*)(?:\*\*(\d+))?', m.group(3))
            ) else m.group(0)
        ),
        expr
    )

    # 3. d**n/dx**n(...) → d_f_x2
    expr = re.sub(
        r'd\*\*(\d+)\s*/\s*(d[a-zA-Z_]\w*\*\*\d+(?:\s*d[a-zA-Z_]\w*\*\*\d+)*)\s*\(\s*([a-zA-Z_]\w*)\s*\)',
        lambda m: (
            f'd_{m.group(3)}_' + power_suffix([
                (v.group(1), v.group(2))
                for v in re.finditer(r'd([a-zA-Z_]\w*)\*\*(\d+)', m.group(2))
            ])
            if int(m.group(1)) == sum(
                int(v.group(2)) for v in re.finditer(r'd([a-zA-Z_]\w*)\*\*(\d+)', m.group(2))
            ) else m.group(0)
        ),
        expr
    )

    # 4. d/dx(f) → d_f_x
    expr = re.sub(r'd\s*/\s*d([a-zA-Z_]\w*)\s*\(\s*([a-zA-Z_]\w*)\s*\)', r'd_\2_\1', expr)

    # 5. dy/dx → d_y_x
    expr = re.sub(r'\bd([a-zA-Z_]\w*)\s*/\s*d([a-zA-Z_]\w*)', r'd_\1_\2', expr)

    # 6. ∂/∂x(f) → del_f_x
    expr = re.sub(r'∂/\s*∂([a-zA-Z_]\w*)\s*\(\s*([a-zA-Z_]\w*)\s*\)', r'del_\2_\1', expr)

    # 7. ∂f/∂x → del_f_x
    expr = re.sub(r'∂\s*([a-zA-Z_]\w*)\s*/\s*∂\s*([a-zA-Z_]\w*)', r'del_\1_\2', expr)

    # 8. ∂**n f / ∂x**n [∂y**m] → del_f_x2_y
    expr = re.sub(
        r'∂\*\*(\d+)\s+([a-zA-Z_]\w*)\s*/\s*(∂[a-zA-Z_]\w*(?:\*\*\d+)?(?:\s+∂[a-zA-Z_]\w*(?:\*\*\d+)?)*)',
        lambda m: (
            f'del_{m.group(2)}_' + power_suffix([
                (v.group(1), v.group(2) or "1")
                for v in re.finditer(r'∂([a-zA-Z_]\w*)(?:\*\*(\d+))?', m.group(3))
            ])
            if int(m.group(1)) == sum(
                int(v.group(2) or "1") for v in re.finditer(r'∂([a-zA-Z_]\w*)(?:\*\*(\d+))?', m.group(3))
            ) else m.group(0)
        ),
        expr
    )

    # 9. partial**n(f)/partial(x)**n → del_f_x2
    expr = re.sub(
        r'partial\*\*(\d+)\s*\(\s*([a-zA-Z_]\w*)\s*\)\s*/\s*(partial\(\s*[a-zA-Z_]\w*\s*\)(?:\*\*\d+)?(?:\s*partial\(\s*[a-zA-Z_]\w*\s*\)(?:\*\*\d+)?)*)',
        lambda m: (
            f'del_{m.group(2)}_' + power_suffix([
                (v.group(1), v.group(2) or "1")
                for v in re.finditer(r'partial\(\s*([a-zA-Z_]\w*)\s*\)(?:\*\*(\d+))?', m.group(3))
            ])
            if int(m.group(1)) == sum(
                int(v.group(2) or "1") for v in re.finditer(r'partial\(\s*([a-zA-Z_]\w*)\s*\)(?:\*\*(\d+))?', m.group(3))
            ) else m.group(0)
        ),
        expr
    )

    # 10. partial(f)/partial(x) → del_f_x
    expr = re.sub(r'partial\(\s*([a-zA-Z_]\w*)\s*\)/partial\(\s*([a-zA-Z_]\w*)\s*\)', r'del_\1_\2', expr)

    # 11. D/Dx(f) → (del_f_t+dot(v,grad(f)))
    expr = re.sub(r'D/D([a-zA-Z_]\w*)\(\s*([a-zA-Z_]\w*)\s*\)', r'D_\2_\1', expr)

    # 12. Df/Dx → (del_f_t+dot(v,grad(f)))
    expr = re.sub(r'\bD([a-zA-Z_]\w*)/D([a-zA-Z_]\w*)\b', r'D_\1_\2', expr)

    return expr.replace(" ", "")

def convert_dot_products(expr: str) -> str:
    # Step 0: a.dot(b) → dot(a,b), a.cross(b) → cross(a,b)
    expr = re.sub(r'\b([a-zA-Z_]\w*)\.dot\(([^()]+)\)', r'dot(\1,\2)', expr)
    expr = re.sub(r'\b([a-zA-Z_]\w*)\.cross\(([^()]+)\)', r'cross(\1,\2)', expr)

    # Step 1: chained dot/cross (e.g. a.b.c → dot(dot(a,b),c))
    def convert_chain(m):
        parts = re.split(r'[.·⋅×]', m.group(0))
        result = parts[0]
        for p in parts[1:]:
            result = f'dot({result},{p})'
        return result

    expr = re.sub(r'\b(?:[a-zA-Z_]\w*[.·⋅×]){2,}[a-zA-Z_]\w*\b', convert_chain, expr)

    # Step 2: single product (a.b, a×b, a·b)
    def convert_single(m):
        full = m.group(0)
        a = m.group(1)
        op = m.group(2)
        b = m.group(3)

        # 除外リスト：.dot や .sin など（最初に処理済みならもう不要でもよい）
        if full in ['np.dot', 'math.sin']:
            return full

        if op in ['.', '·', '⋅']:
            return f'dot({a},{b})'
        elif op == '×':
            return f'cross({a},{b})'
        else:
            return full

    expr = re.sub(r'\b([a-zA-Z_]\w*)([.·⋅×])([a-zA-Z_]\w*)\b', convert_single, expr)
    # Step 3: t·(cross(p,q)) → dot(t, cross(p,q))
    expr = re.sub(r'\b([a-zA-Z_]\w*)[·⋅]\(((?:[^()]|\([^()]*\))*)\)', r'dot(\1, \2)', expr)
    return expr


def convert_other_re(expr: str,params) -> str:

    # 以下は既存の置換処理
    # Gradient
    expr = re.sub(r'\b(∇f|gradient\(([^()]+)\)|grad\(([^()]+)\))\b',
                  lambda m: f'grad({m.group(2) or m.group(3)})', expr)

    # Divergence
    expr = re.sub(r'\b(∇·([a-zA-Z_]\w*)|divergence\(([^()]+)\)|div\(([^()]+)\))',
                  lambda m: f'divg({m.group(2) or m.group(3) or m.group(4)})', expr)

    # Curl
    expr = re.sub(r'\b(∇×([a-zA-Z_]\w*)|curl\(([^()]+)\)|rot\(([^()]+)\))\b',
                  lambda m: f'rot({m.group(2) or m.group(3) or m.group(4)})', expr)

    # Infinity
    expr = re.sub(r'\b(Infinity|infinity|∞|Inf|inf|Infty|infty)\b', 'oo', expr)
    expr = re.sub(r'\boo\b', 'oo', expr)

    # Vector
    expr = re.sub(r'\\vec\s*\{\{([a-zA-Z_]\w*)\}\}', r'vec(\1)', expr)
    expr = re.sub(r'\bvector\(([^()]+)\)|\b([a-zA-Z_]\w*)\.vector\(\)|\bMatrix\(([^()]+)\)',
                  lambda m: f'vec({m.group(1) or m.group(2) or m.group(3)})', expr)

    # Unit vector / hat
    # \hat{{a}}, unit(...), normalize(...), a_hat, and Unicode â form
    expr = re.sub(
        r'(\\hat\{\{([a-zA-Z_]\w*)\}\}|unit\(([^()]+)\)|normalize\(([^()]+)\)|([a-zA-Z_]\w*)_hat)',
        lambda m: f'hat({m.group(2) or m.group(3) or m.group(4) or m.group(5)})',
        expr
    )

    # Specific Unicode characters with combining hat: î, ĵ, etc.
    expr = re.sub(r'([eijknrzθ])\u0302', r'hat(\1)', expr)


    # Exponential
    expr = re.sub(r'\be\*\*\s*([a-zA-Z_]\w*)\b', r'exp(\1)', expr)
    expr = re.sub(r'\bexponential\(([^()]+)\)', r'exp(\1)', expr)
    expr = re.sub(r'\bexp\(([^()]+)\)', r'exp(\1)', expr)

    expr = convert_dot_products(expr)

    return expr

def convert_integral_re(expr: str, params) -> str:
    prev = None
    while expr != prev:
        prev = expr

        # ---------- 1. Closed (circular) integrals ----------
        expr = re.sub(
            r"∮\s*"
            r"(?:_\{([a-zA-Z0-9_+\-*/^()\[\]{}]+)\}"
            r"|_\(([a-zA-Z0-9_+\-*/^()\[\]{}]+)\)"
            r"|_([a-zA-Z0-9_+\-*/^()\[\]{}]+))?"
            r"\s*([a-zA-Z0-9_+\-*/^()\[\]{}, ]+?)"
            r"(?:(?<=\s)|(?<=[^a-zA-Z0-9+\-*/_]))d\s*([a-zA-Z])",
            lambda m: f"ointg({m.group(4).strip()},{m.group(5)})",
            expr
        )
        expr = re.sub(
            r"oint\s*\(\s*[\(\{\[]\s*([^,]+?)\s*,\s*([^,\]\)\}]+?)\s*[\)\}\]]\s*\)",
            lambda m: f"ointg({m.group(1)},{m.group(2)})",
            expr
        )
        expr = re.sub(
            r"int\s*\(\s*([^,]+?)\s*,\s*[\(\{\[]\s*([^,]+?)\s*,\s*([^,\]\)\}]+?)\s*[\)\}\]]\s*,\s*circular\s*=\s*True\s*\)",
            lambda m: f"ointg({m.group(1)},{m.group(2)})",
            expr
        )
        expr = re.sub(
            r"integrate\s*\(\s*([^,]+?)\s*,\s*[\(\{\[]\s*([^,]+?)\s*,\s*([^,\]\)\}]+?)\s*[\)\}\]]\s*,\s*circular\s*=\s*True\s*\)",
            lambda m: f"ointg({m.group(1)},{m.group(2)})",
            expr
        )
        expr = re.sub(
            r"Integral\s*\(\s*((?:[^(){}[\]]|\([^()]*\))+?)\s*,\s*\(\s*([^\s,()]+)\s*,[^)]+\)\s*,\s*circular\s*=\s*True\s*\)",
            lambda m: f"ointg({m.group(1).strip()},{m.group(2).strip()})",
            expr
        )

        # ---------- 2. Definite integrals ----------
        expr = re.sub(
            r"∫_"
            r"(?:\{([^\}]+)\}|([a-zA-Z0-9_+\-*/^()]+))"
            r"\^"
            r"(?:\{([^\}]+)\}|([a-zA-Z0-9_+\-*/^()]+))"
            r"\s+([a-zA-Z0-9_+\-*/^()]+)\s*d\s*([a-zA-Z])",
            lambda m: f"intg({m.group(5)},({m.group(6)},{m.group(1) or m.group(2)},{m.group(3) or m.group(4)}))",
            expr
        )
        expr = re.sub(
            r"int\(([^,]+),\s*\(([^,]+),\s*([^,]+),\s*([^,]+)\)\)",
            lambda m: f"intg({m.group(1)},({m.group(2)},{m.group(3)},{m.group(4)}))",
            expr
        )
        expr = re.sub(
            r"int_([a-zA-Z0-9_+\-*/^()]+)\^([a-zA-Z0-9_+\-*/^()]+)\s*\(\s*([a-zA-Z0-9_+\-*/^()]+)\s*,\s*([a-zA-Z])\s*\)",
            lambda m: f"intg({m.group(3)},({m.group(4)},{m.group(1)},{m.group(2)}))",
            expr
        )
        expr = re.sub(
            r"int_([a-zA-Z0-9_+\-*/^()]+)\^([a-zA-Z0-9_+\-*/^()]+)\s*\(\s*([a-zA-Z0-9_+\-*/^()]+)\s*\)\s*d\s*([a-zA-Z])",
            lambda m: f"intg({m.group(3)},({m.group(4)},{m.group(1)},{m.group(2)}))",
            expr
        )
        expr = re.sub(
            r"int_([a-zA-Z0-9_+\-*/^()]+)\^([a-zA-Z0-9_+\-*/^()]+)\s+([a-zA-Z0-9_+\-*/^()]+)\s*d\s*([a-zA-Z])",
            lambda m: f"intg({m.group(3)},({m.group(4)},{m.group(1)},{m.group(2)}))",
            expr
        )

        # ---------- 3. Lower-limit-only integrals ----------
        expr = re.sub(
            r"∫(?:_\{([^\}]+)\}|_\(([^\)]+)\)|_([a-zA-Z0-9_]+))?"
            r"\s*([a-zA-Z0-9_+\-*/^(){}\[\]]+)\s*d\s*([a-zA-Z])",
            lambda m: f"intg({m.group(4)},{m.group(5)})",
            expr
        )
        expr = re.sub(
            r"int\s*\(\s*[\(\{\[]\s*([^,]+?)\s*,\s*([^,\]\)\}]+?)\s*[\)\}\]]\s*\)",
            lambda m: f"intg({m.group(1)},{m.group(2)})",
            expr
        )
        expr = re.sub(
            r"int_([a-zA-Z0-9_+\-*/^()]+)\s*\(\s*([a-zA-Z0-9_+\-*/^()]+)\s*,\s*([a-zA-Z])\s*\)",
            lambda m: f"intg({m.group(2)},{m.group(3)})",
            expr
        )

        # ---------- 4. Standard single integrals ----------
        expr = re.sub(
            r"∫\s*([^\s∫]+?)\s*d\s*([a-zA-Z])",
            lambda m: f"intg({m.group(1)},{m.group(2)})",
            expr
        )
        expr = re.sub(
            r"int\(([^,]+),\s*([^)]+)\)",
            lambda m: f"intg({m.group(1)},{m.group(2)})",
            expr
        )
        expr = re.sub(
            r"integrate\(([^,]+),\s*([^)]+)\)",
            lambda m: f"intg({m.group(1)},{m.group(2)})",
            expr
        )
        expr = re.sub(
            r"Integral\(([^,]+),\s*([^)]+)\)",
            lambda m: f"intg({m.group(1)},{m.group(2)})",
            expr
        )

    return expr