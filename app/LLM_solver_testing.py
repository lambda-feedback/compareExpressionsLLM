from LLM_solver import LLM_solve

Q_list = [
    ["Does water freeze at 0 degrees Celsius?", "BOOLEAN"],
    ["Is an electron heavier than a proton?", "BOOLEAN"],
    ["Solve for x. x^2 - 4 = 0", "EXPRESSION"],
    ["d/dx (x^3)", "EXPRESSION"],
    ["Write the unitary 2x2 matrix", "MATRIX"],
    ["Transpose [[1,2,3],[4,5,6]]", "MATRIX"],
    ["What is the largest planet in our solar system? (a) Earth (b) Jupiter (c) Saturn (d) Venus", "MULTIPLE_CHOICE"],
    ["What is the chemical symbol for oxygen? (a) O (b) H (c) C (d) N", "MULTIPLE_CHOICE"],
    ["5 + 7", "NUMBER"],
    ["Square root of 9", "NUMBER"],
    ["What is the acceleration due to gravity on Earth in m/s^2?", "NUMERIC_UNITS"],
    ["Speed of light in vacuum in m/s?", "NUMERIC_UNITS"],
    ["What is the satellite of Earth?", "TEXT"],
    ["What is the chemical formula for water?", "TEXT"],
]

Q_list_fmx3 = [
    ["Use elementary geometry to derive the rotation matrix that converts the Cartesian vector components (u₁, u₂, u₃) into cylindrical polar components (Vᵣ, Vθ, Vz). Write the 3x3 linear vector component transformation matrix.",
     "MATRIX", "$\begin{pmatrix}V_{r} \\ V_{\theta} \\ V_{z} \end{pmatrix}=$", "$\begin{pmatrix}u_{1} \\ u_{2} \\ u_{3} \end{pmatrix}$"],
]

for q, t, pre, post in Q_list_fmx3:
    answer = LLM_solve(q, t, pre, post)
    print(answer)