import re
from collections import Counter

def convert_diff_re(expr: str, params) -> str:
    s = expr

    # 0) 物質導関数（順番が重要）
    # D/Dt(y) → (smart_derivative(y,t,1)+smart_dot(Gradient(y),u_vec))
    s = re.sub(
        r'\bD\s*/\s*D([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)',
        r'(smart_derivative(\2,\1,1)+smart_dot(Gradient(\2),u_vec))',
        s
    )
    # Dy/Dt → (smart_derivative(y,t,1)+smart_dot(Gradient(y),u_vec))
    s = re.sub(
        r'\bD([A-Za-z_]\w*)\s*/\s*D([A-Za-z_]\w*)\b',
        r'(smart_derivative(\1,\2,1)+smart_dot(Gradient(\1),u_vec))',
        s
    )

    # 1) diff(y,x[,x...]) → smart_derivative(y,x,n)  (同一変数の反復のみ対応)
    def repl_diff(m):
        func = m.group(1)
        args = [v.strip() for v in m.group(2).split(',')]
        if not args:
            return m.group(0)
        # すべて同じ変数のときだけ次数にまとめる
        if all(a == args[0] for a in args):
            return f"smart_derivative({func},{args[0]},{len(args)})"
        # 混在はルール外 → そのまま
        return m.group(0)

    s = re.sub(
        r'\bdiff\(\s*([A-Za-z_]\w*)\s*,\s*((?:[A-Za-z_]\w*\s*,\s*)*[A-Za-z_]\w*)\s*\)',
        repl_diff,
        s
    )

    # 2) d/dx(y) → smart_derivative(y,x,1)
    s = re.sub(
        r'\bdel\s*/\s*del([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)',
        r'smart_derivative(\2,\1,1)',
        s
    )

    # 3) dy/dx → smart_derivative(y,x,1)
    s = re.sub(
        r'\bdel([A-Za-z_]\w*)\s*/\s*del([A-Za-z_]\w*)\b',
        r'smart_derivative(\1,\2,1)',
        s
    )

    # 4) d**n y / d x**n → smart_derivative(y,x,n)
    s = re.sub(
        r'\bdel\*\*(\d+)\s*([A-Za-z_]\w*)\s*/\s*del([A-Za-z_]\w*)\s*\*\*\s*\1\b',
        r'smart_derivative(\2,\3,\1)',
        s
    )

    # 2) d/dx(y) → smart_derivative(y,x,1)
    s = re.sub(
        r'\bd\s*/\s*d([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)',
        r'smart_derivative(\2,\1,1)',
        s
    )

    # 3) dy/dx → smart_derivative(y,x,1)
    s = re.sub(
        r'\bd([A-Za-z_]\w*)\s*/\s*d([A-Za-z_]\w*)\b',
        r'smart_derivative(\1,\2,1)',
        s
    )

    # 4) d**n y / d x**n → smart_derivative(y,x,n)
    s = re.sub(
        r'\bd\*\*(\d+)\s*([A-Za-z_]\w*)\s*/\s*d([A-Za-z_]\w*)\s*\*\*\s*\1\b',
        r'smart_derivative(\2,\3,\1)',
        s
    )

    # 5) ∂/∂x (y) → smart_derivative(y,x,1)
    s = re.sub(
        r'∂\s*/\s*∂\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)',
        r'smart_derivative(\2,\1,1)',
        s
    )

    # 6) ∂y/∂x → smart_derivative(y,x,1)
    s = re.sub(
        r'∂\s*([A-Za-z_]\w*)\s*/\s*∂\s*([A-Za-z_]\w*)',
        r'smart_derivative(\1,\2,1)',
        s
    )

    # 7) ∂**n y / ∂x**n → smart_derivative(y,x,n)
    s = re.sub(
        r'∂\*\*(\d+)\s*([A-Za-z_]\w*)\s*/\s*∂([A-Za-z_]\w*)\s*\*\*\s*\1\b',
        r'smart_derivative(\2,\3,\1)',
        s
    )

    # 8) partial**n(y)/partial(x)**n → smart_derivative(y,x,n)
    s = re.sub(
        r'partial\*\*(\d+)\s*\(\s*([A-Za-z_]\w*)\s*\)\s*/\s*partial\(\s*([A-Za-z_]\w*)\s*\)\s*\*\*\s*\1\b',
        r'smart_derivative(\2,\3,\1)',
        s
    )

    # 9) partial(y)/partial(x) → smart_derivative(y,x,1)
    s = re.sub(
        r'partial\(\s*([A-Za-z_]\w*)\s*\)\s*/\s*partial\(\s*([A-Za-z_]\w*)\s*\)',
        r'smart_derivative(\1,\2,1)',
        s
    )

    return s.replace(" ", "")

def convert_integral_re(expr: str, params) -> str:
    s = expr

    # 内側の int も全部 Integral にする保険
    s = re.sub(r'\bint\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', r'Integral(\1,\2)', s)
    s = re.sub(r'\bintegrate\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', r'Integral(\1,\2)', s)

    # 0) 閉曲線（circular=True/ ∮ ... ）→ Integral(f,(x,0,2*pi))
    #   int(f,(x,a),circular=True) / integrate / Integral
    s = re.sub(
        r'\b(?:int|integrate|Integral)\s*\(\s*([^,]+)\s*,\s*\(\s*([A-Za-z_]\w*)\s*,\s*[^,)]+\)\s*,\s*circular\s*=\s*True\s*\)',
        r'Integral(\1,(\2,0,2*pi))',
        s
    )
    s = re.sub(
        r'∮\s*([A-Za-z0-9_+\-*/^()]+)\s*d\s*([A-Za-z_]\w*)',
        r'Integral(\1,(\2,0,2*pi))',
        s
    )

    # 1) 有限区間の定積分 → Integral(f,(x,a,b))
    # ∫_a^b f dx
    s = re.sub(
        r'∫_\s*(?:\{([^}]+)\}|([^\^ ]+))\s*\^\s*(?:\{([^}]+)\}|([^\s]+))\s*([A-Za-z0-9_+\-*/^()]+)\s*d\s*([A-Za-z_]\w*)',
        lambda m: f"Integral({m.group(5)},({m.group(6)},{m.group(1) or m.group(2)},{m.group(3) or m.group(4)}))",
        s
    )
    # int(f,(x,a,b)) / integrate / Integral
    s = re.sub(
        r'\b(?:int|integrate|Integral)\s*\(\s*([^,]+)\s*,\s*\(\s*([A-Za-z_]\w*)\s*,\s*([^,]+)\s*,\s*([^)]+)\)\s*\)',
        r'Integral(\1,(\2,\3,\4))',
        s
    )
    # int_a^b (f, x) / int_a^b f dx
    s = re.sub(
        r'\bint_([^\^]+)\^([^(]+)\s*\(\s*([^,]+)\s*,\s*([A-Za-z_]\w*)\s*\)',
        r'Integral(\3,(\4,\1,\2))',
        s
    )
    s = re.sub(
        r'\bint_([^\^ ]+)\^([^\s]+)\s+([A-Za-z0-9_+\-*/^()]+)\s*d\s*([A-Za-z_]\w*)',
        r'Integral(\3,(\4,\1,\2))',
        s
    )

    # 2) 下限のみ（上限なし）→ Integral(f,x) に落とす
    s = re.sub(
        r'∫(?:_\{[^}]+\}|_\([^)]+\)|_[A-Za-z0-9_]+)?\s*([A-Za-z0-9_+\-*/^(){}\[\]]+)\s*d\s*([A-Za-z_]\w*)',
        r'Integral(\1,\2)',
        s
    )
    s = re.sub(
        r'\bint\s*\(\s*\(\s*([^,]+)\s*,\s*([A-Za-z_]\w*)\s*\)\s*\)',
        r'Integral(\1,\2)',
        s
    )
    s = re.sub(
        r'\bint_([A-Za-z0-9_+\-*/^()]+)\s*\(\s*([A-Za-z0-9_+\-*/^()]+)\s*,\s*([A-Za-z_]\w*)\s*\)',
        r'Integral(\2,\3)',
        s
    )

    # 3) 単純な単積分 → Integral(f,x)
    s = re.sub(
        r'∫\s*([^\s∫]+)\s*d\s*([A-Za-z_]\w*)',
        r'Integral(\1,\2)',
        s
    )
    s = re.sub(
        r'\bint\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
        r'Integral(\1,\2)',
        s
    )
    s = re.sub(
        r'\bintegrate\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
        r'Integral(\1,\2)',
        s
    )
    s = re.sub(
        r'\bIntegral\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
        r'Integral(\1,\2)',
        s
    )

    # 4) 多重積分： integrate/Integral(f, x, y[, z ...]) → ネスト
    #   粗いが有効：f, 変数たち を取り出して右からネスト
    def nest_multi(m):
        inner = m.group(1).strip()
        rest = [v.strip() for v in m.group(2).split(',')]
        # rest は変数列（(x) or (x,a,b) で来る可能性があるので括弧を保つ）
        for part in rest:
            inner = f"Integral({inner},{part})"
        return inner

    s = re.sub(
        r'\b(?:integrate|Integral)\s*\(\s*([^,]+)\s*,\s*((?:\([^)]+\)|[A-Za-z_]\w*)(?:\s*,\s*(?:\([^)]+\)|[A-Za-z_]\w*))+)\s*\)',
        nest_multi,
        s
    )

    return s.replace(" ", "")

def convert_other_re(expr: str, params) -> str:
    s = expr

    # Gradient: [∇f, gradient(f), grad(f), nabla(f)] -> Gradient(f)
    s = re.sub(
        r'(?:∇\s*\(\s*([A-Za-z_]\w*)\s*\)|∇\s*([A-Za-z_]\w*)|gradient\(\s*([A-Za-z_]\w*)\s*\)|grad\(\s*([A-Za-z_]\w*)\s*\)|nabla\(\s*([A-Za-z_]\w*)\s*\))',
        lambda m: f"Gradient({m.group(1) or m.group(2) or m.group(3) or m.group(4) or m.group(5)})",
        s
    )

    # Divergence: [∇·f, div(f), divergence(f), nabla·f] -> Divergence(f)
    s = re.sub(
        r'\b(?:∇·\s*([A-Za-z_]\w*)|divergence\(([^()]+)\)|div\(([^()]+)\)|nabla·\s*([A-Za-z_]\w*))',
        lambda m: f"Divergence({m.group(1) or m.group(2) or m.group(3) or m.group(4)})",
        s
    )

    # Curl: [∇×f, curl(f), rot(f), nablaxf, ∇xf] -> Curl(f)
    s = re.sub(
        r'\b(?:∇×\s*([A-Za-z_]\w*)|curl\(([^()]+)\)|rot\(([^()]+)\)|nablax([A-Za-z_]\w*)|∇x([A-Za-z_]\w*))\b',
        lambda m: f"Curl({m.group(1) or m.group(2) or m.group(3) or m.group(4) or m.group(5)})",
        s
    )

    # Infinity: -> inf
    s = re.sub(r'\b(?:Infinity|infinity|∞|Inf|Infty|infty|oo)\b', 'inf', s)

    # Dot/Cross (先に a.b / a·b / a⋅b / dot(a,b) → smart_dot )
    # メソッド呼び出し a.dot(b)
    s = re.sub(r'\b([A-Za-z_]\w*)\.dot\(([^()]+)\)', r'smart_dot(\1,\2)', s)
    # 関数 dot(a,b)
    s = re.sub(r'\bdot\(\s*([^,]+)\s*,\s*([^)]+)\)', r'smart_dot(\1,\2)', s)
    # 中置 a·b / a⋅b
    s = re.sub(r'\b([A-Za-z_]\w*)[·⋅]([A-Za-z_]\w*)\b', r'smart_dot(\1,\2)', s)

    # Cross: [a×b, cross(a,b)] -> a.cross(b)
    s = re.sub(r'\bcross\(\s*([^,]+)\s*,\s*([^)]+)\)', r'\1.cross(\2)', s)
    s = re.sub(r'\b([A-Za-z_]\w*)×([A-Za-z_]\w*)\b', r'\1.cross(\2)', s)

    # Vector: [\vec{a}, vector(a), a.vector(), Matrix(a)] -> Matrix(a)
    s = re.sub(r'\\vec\s*\{\s*([A-Za-z_]\w*)\s*\}', r'Matrix(\1)', s)
    s = re.sub(r'\bvector\(([^()]+)\)', r'Matrix(\1)', s)
    s = re.sub(r'\b([A-Za-z_]\w*)\.vector\(\)', r'Matrix(\1)', s)
    # Matrix(a) はそのままでOK（既に目標形）

    # Unit / hat: [â, \hat{a}, unit(a), normalize(a), hat(a)] -> a_hat
    s = re.sub(r'\\hat\{\s*([A-Za-z_]\w*)\s*\}', r'\1_hat', s)
    s = re.sub(r'\bunit\(([^()]+)\)', r'\1_hat', s)
    s = re.sub(r'\bnormalize\(([^()]+)\)', r'\1_hat', s)
    s = re.sub(r'\bhat\(([^()]+)\)', r'\1_hat', s)
    # 一般的な合成ハット（Unicode結合）: â → a_hat
    s = re.sub(r'([A-Za-z_]\w*)\u0302', r'\1_hat', s)

    # Exponential: [exp(x), e**x, exponential(x)] -> exp(x)
    s = re.sub(r'\be\*\*\s*([A-Za-z_]\w*)\b', r'exp(\1)', s)
    s = re.sub(r'\bexponential\(([^()]+)\)', r'exp(\1)', s)
    # 既に exp(...) のものはそのまま（正規化だけ）
    s = re.sub(r'\bexp\(([^()]+)\)', r'exp(\1)', s)

    return s.replace(" ", "")