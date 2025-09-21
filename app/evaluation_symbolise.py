def is_valid_operand(s):
    s = s.strip()
    if s == "":
        return False
    return s[0] not in "+-*/=)"

def symbolise_by_operators(expr1, expr2):
    symbol_map = {}
    counter = [0]

    def get_sym(subexpr_str):
        if subexpr_str in {"0"}:
            return subexpr_str
        subexpr_str = subexpr_str.strip()
        if subexpr_str not in symbol_map:
            symbol_map[subexpr_str] = f"s{counter[0]}"
            counter[0] += 1
        return symbol_map[subexpr_str]

    def _rec(s):
        s = s.strip()
        if s in symbol_map:
            return symbol_map[s]

        # 1. Enclosed with parentheses
        if s.startswith("(") and s.endswith(")") and is_matching_parens(s):
            inner = s[1:-1].strip()
            # Check if inner is already symbolised
            if s in symbol_map:
                return get_sym(s)
            return f"({_rec(inner)})"

        # 2. Equality
        if "=" in s and "==" not in s:
            parts = split_top_level(s, "=")
            if len(parts) == 2:
                left, right = parts
                # Strip and remove outer parentheses if they match
                left = left.strip()
                right = right.strip()
                if left.startswith("(") and left.endswith(")") and is_matching_parens(left):
                    left = left[1:-1]
                if right.startswith("(") and right.endswith(")") and is_matching_parens(right):
                    right = right[1:-1]
                return f"{_rec(left)} = {_rec(right)}"
        
        # 3. Exponentiation
        parts = split_top_level(s, "**")
        if len(parts) == 2:
            base, exp = parts
            return f"{_rec(base)}**{_rec(exp)}"

        # 4. Multiplication and Division
        for op in ["*", "/"]:
            parts = split_top_level(s, op)
            if len(parts) > 1:
                joined = f" {op} ".join([_rec(p) for p in parts])
                return f"{joined}"

        if s.startswith("-") and is_valid_operand(s[1:]):
            return f"-{_rec(s[1:])}"
        
        # 5. Addition and Subtraction
        for op in ["+", "-"]:
            parts = split_top_level(s, op)
            if len(parts) > 1:
                joined = f" {op} ".join([_rec(p) for p in parts])
                return f"{joined}"

        # 6. If nothing matches, symbolise the whole thing
        return get_sym(s)

    return _rec(expr1), _rec(expr2), symbol_map


def is_matching_parens(s):
    depth = 0
    for i, c in enumerate(s):
        if c == "(": depth += 1
        elif c == ")": depth -= 1
        if depth == 0 and i < len(s) - 1:
            return False
        if depth < 0:
            return False
    return depth == 0



def split_top_level(s, sep):
    parts = []
    depth = 0
    last = 0
    i = 0
    while i < len(s):
        if s[i] == "(": depth += 1
        elif s[i] == ")": depth -= 1
        elif s[i:i+len(sep)] == sep and depth == 0:
            parts.append(s[last:i])
            last = i + len(sep)
            i += len(sep)
            continue
        i += 1
    parts.append(s[last:])
    return parts

# e1 = "(a+b)*(c+d)"
# e2 = "a*c+a*d+b*c+b*d"

# t1, t2, symbol_map = symbolise_by_operators(e1, e2)
# print("Expr1:", t1)
# print("Expr2:", t2)
# print(symbol_map)
# test_cases = [
#     # 1. Simple addition
#     ("a + b", "b + c"),

#     # 2. Function expressions
#     ("sin(x) + cos(y)", "cos(y) + tan(z)"),

#     # 3. Nested parentheses
#     ("x * (y + z)", "(y + x) * y"),

#     # 4. Whitespace & parenthesis
#     ("a*(b+c)", "(b+c) * a"),

#     # 5. Equation with shared LHS
#     ("a + b = c", "a + b = d"),

#     # 7. Fully different terms
#     ("foo + bar", "baz + qux"),

#     # 8. Commutativity test
#     ("m ** (n + p)", "m**n+p"),
#     ("(m+n)=-p","(m+n+p)=0"),
# ]

# for i, (expr1, expr2) in enumerate(test_cases, 1):
#     s1, s2, symbol_map = symbolise_by_operators(expr1, expr2)
#     print(f"--- Test Case {i} ---")
#     print("Expr1:", expr1)
#     print("Expr2:", expr2)
#     print("→ Symbolised Expr1:", s1)
#     print("→ Symbolised Expr2:", s2)
#     print(symbol_map)
#     print()
