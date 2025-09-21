preview_test_cases = [
    # # Basic arithmetic
    # ["2+2", r"$2+2$", True],
    # ["x^2+2x+1", r"$x^{2}+2x+1$", True],
    # ["(x+1)**2", r"$ \left(x+1 \right)^{2}$", True],
    # ["x**2 - 1", r"$x^{2}-1$", True],
    # ["(x-1)(x+1)", r"$ \left(x-1 \right) \left(x+1 \right)$", True],

    # # Trig
    # [r"\sin(x)^2+\cos(x)^2", r"$ \sin \left(x \right)^{2}+ \cos \left(x \right)^{2}$", True],
    # ["sin(x) + cos(x)", r"$ \sin \left(x \right)+ \cos \left(x \right)$", True],
    # ["sin(x) * cos(x)", r"$ \sin \left(x \right) \cos \left(x \right)$", True],
    # ["tan(x)", r"$ \tan \left(x \right)$", True],

    # # Exp and Log
    # ["exp(x)*exp(y)", r"$ \exp \left(x \right) \exp \left(y \right)$", True],
    # ["exp(x+y)", r"$ \exp \left(x+y \right)$", True],
    # ["log(x*y)", r"$ \log \left(x y \right)$", True],
    # ["log(x)+log(y)", r"$ \log \left(x \right)+ \log \left(y \right)$", True],

    # Calculus
    ["dy/dx", r"$\frac{dy}{dx}$", True],
    ["diff(y,x)", r"$\frac{dy}{dx}$", True],
    ["d^3y/dx^3", r"$\frac{d^{3}y}{dx^{3}}$", True],
    ["partial z/partial x", r"$\frac{\partial z}{\partial x}$", True],
    ["∂z/∂x", r"$\frac{\partial z}{\partial x}$", True],
    ["∂^2z/∂x^2", r"$\frac{\partial^{2}z}{\partial x^{2}}$", True],
    ["∂^2z/∂x∂y", r"$\frac{\partial^{2}z}{\partial x \partial y}$", True],
    ["∫ x dx", r"$\int x \, dx$", True],
    ["∫f dx", r"$\int f \, dx$", True],
    ["int f dx", r"$\int f \, dx$", True],
    ["oint f dx", r"$ \oint f \, dx$", True],
    ["int_0^1 f dx", r"$ \int_{0}^{1} f \, dx$", True],
    [r"\int\int f dx dy", r"$ \int \int f \, dx \, dy$", True],

    # # Others
    # ["infty", r"$\infty$", True],
    # ["Infinity", r"$\infty$", True],
    # ["i", r"$i$", True],
    # ["sqrt(-1)", r"$ \sqrt{-1}$", True],
    
    ["nabla", r"$ \nabla$", True],
    ["a dot b", r"$a \cdot b$", True],
    ["a cross b", r"$a \times b$", True],
]
