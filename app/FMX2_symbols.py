from sympy import symbols, Function, Matrix, Expr, Derivative
from sympy.matrices import MatrixBase

class Fluids:
    def __init__(self):
        # 座標
        self.x, self.y, self.z, self.t = symbols('x y z t')
        self.r, self.theta = symbols('r theta')

        # 物理量（スカラー場）
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

        # 定数
        self.a, self.b, self.c, self.h = symbols('a b c h')
        self.m, self.g, self.mu, self.nu, self.R, self.c_p, self.c_v, self.kappa = symbols(
            'm g mu nu R c_p c_v kappa', positive=True
        )

        # 速度ベクトル（Matrixベース）
        self.u_vec = Matrix([[self.u],[self.v],[self.w]])
        self.x_vec = Matrix([[self.x],[self.y],[self.z]])

        # 勾配テンソル（成分ごとの微分）
        self.grad_u = self.Gradient(self.u_vec)

        # 円筒座標系の速度（Matrixベース）
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
        """スカラーかベクトルに応じて勾配 or 勾配テンソルを返す"""
        if isinstance(f, Expr):
            return Matrix([
                [f.diff(self.x)],
                [f.diff(self.y)],
                [f.diff(self.z)]
            ])

        elif isinstance(f, MatrixBase) and f.shape == (3, 1):
            return Matrix([
                [f[0].diff(self.x), f[0].diff(self.y), f[0].diff(self.z)],
                [f[1].diff(self.x), f[1].diff(self.y), f[1].diff(self.z)],
                [f[2].diff(self.x), f[2].diff(self.y), f[2].diff(self.z)]
            ])

        else:
            raise TypeError("Gradient() expects a scalar Expr or a 3x1 Matrix.")

    def Divergence(self, vec):
        """ベクトルの発散（スカラー）"""
        if not (isinstance(vec, MatrixBase) and vec.shape == (3, 1)):
            raise TypeError("Divergence expects a 3x1 vector Matrix.")
        return vec[0].diff(self.x) + vec[1].diff(self.y) + vec[2].diff(self.z)

    def Curl(self, vec):
        """ベクトルの回転（3x1ベクトル）"""
        if not (isinstance(vec, MatrixBase) and vec.shape == (3, 1)):
            raise TypeError("Curl expects a 3x1 vector Matrix.")
        return Matrix([
            [vec[2].diff(self.y) - vec[1].diff(self.z)],
            [vec[0].diff(self.z) - vec[2].diff(self.x)],
            [vec[1].diff(self.x) - vec[0].diff(self.y)]
        ])
    def smart_derivative(self, expr, var):
        # スカラーならそのまま
        if not isinstance(expr, MatrixBase):
            return Derivative(expr, var)
        # ベクトルなら成分ごとに
        return Matrix([[Derivative(expr[i], var)] for i in range(expr.rows)])
