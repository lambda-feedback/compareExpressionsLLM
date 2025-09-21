from sympy import symbols, Function, Matrix, Expr, Derivative
from sympy.matrices import MatrixBase

class Fluids:
    def __init__(self):
        self.x, self.y, self.z, self.t = symbols('x y z t')
        self.r, self.theta = symbols('r theta')
        self.inf = symbols('inf')

        self.u_func = Function('u')
        self.v_func = Function('v')
        self.w_func = Function('w')
        self.T_func = Function('T')
        self.rho_func = Function('rho')
        self.p_func = Function('p')

        self.u = self.u_func(self.x, self.y, self.z, self.t)
        self.v = self.v_func(self.x, self.y, self.z, self.t)
        self.w = self.w_func(self.x, self.y, self.z, self.t)
        self.T = self.T_func(self.x, self.y, self.z, self.t)
        self.rho = self.rho_func(self.x, self.y, self.z, self.t)
        self.p = self.p_func(self.x, self.y, self.z, self.t)

        self.a, self.b, self.c, self.h = symbols('a b c h')
        self.m, self.g, self.mu, self.nu, self.R, self.c_p, self.c_v, self.kappa = symbols(
            'm g mu nu R c_p c_v kappa', positive=True
        )

        self.u_vec = Matrix([[self.u],[self.v],[self.w]])
        self.x_vec = Matrix([[self.x],[self.y],[self.z]])

        self.grad_u = self.Gradient(self.u_vec)

        self.u_r_func = Function('u_r')
        self.u_theta_func = Function('u_theta')
        self.u_z_func = Function('u_z')

        self.u_r = self.u_r_func(self.r, self.theta, self.z, self.t)
        self.u_theta = self.u_theta_func(self.r, self.theta, self.z, self.t)
        self.u_z = self.u_z_func(self.r, self.theta, self.z, self.t)

        self.u_vec_cyl = Matrix([[self.u_r],[self.u_theta],[self.u_z]])

        self.x_hat = Matrix([[1],[0],[0]])
        self.y_hat = Matrix([[0],[1],[0]])
        self.z_hat = Matrix([[0],[0],[1]])
        self.r_hat = Matrix([[1],[0],[0]])
        self.theta_hat = Matrix([[0],[1],[0]])

    def Gradient(self, f):
        if isinstance(f, Expr):
            return Matrix([
                [Derivative(f, self.x)],
                [Derivative(f, self.y)],
                [Derivative(f, self.z)]
            ])

        elif isinstance(f, MatrixBase) and f.shape == (3, 1):
            return Matrix([
                [Derivative(f[0], self.x), Derivative(f[0], self.y), Derivative(f[0], self.z)],
                [Derivative(f[1], self.x), Derivative(f[1], self.y), Derivative(f[1], self.z)],
                [Derivative(f[2], self.x), Derivative(f[2], self.y), Derivative(f[2], self.z)]
            ])

        else:
            raise TypeError("Gradient() expects a scalar Expr or a 3x1 Matrix.")

    def Divergence(self, vec):
        if not (isinstance(vec, MatrixBase) and vec.shape == (3, 1)):
            raise TypeError("Divergence expects a 3x1 vector Matrix.")
        return Derivative(vec[0],(self.x)) + Derivative(vec[1],(self.y)) + Derivative(vec[2],(self.z))

    def Curl(self, vec):
        if not (isinstance(vec, MatrixBase) and vec.shape == (3, 1)):
            raise TypeError("Curl expects a 3x1 vector Matrix.")
        return Matrix([
            [Derivative(vec[2],(self.y)) - Derivative(vec[1],(self.z))],
            [Derivative(vec[0],(self.z)) - Derivative(vec[2],(self.x))],
            [Derivative(vec[1],(self.x)) - Derivative(vec[0],(self.y))]
        ])
    def smart_derivative(self, expr, var, n=1):
        if not isinstance(expr, MatrixBase):
            return Derivative(expr, var, n)
        return Matrix([[Derivative(expr[i], var, n)] for i in range(expr.rows)])
    def smart_dot(self, expr1, expr2):
        if isinstance(expr1, MatrixBase) and isinstance(expr2, MatrixBase):
            if expr1.shape == (3, 1) and expr2.shape == (3, 1):
                return expr1.dot(expr2)
        return expr1 * expr2