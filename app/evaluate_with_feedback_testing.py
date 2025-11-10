from evaluate_with_feedback import eval_with_feedback

question_markdown = """
Unless otherwise stated, assume standard atmosphere values of $\rho=1.225 \mathrm{~kg} / \mathrm{m}^{3}, \mu=1.79 \times 10^{-5} \mathrm{~kg} /(\mathrm{ms}), R=287.1 \mathrm{~J} /(\mathrm{kgK})$ and $\gamma=1.4$.

A model of the real, separated flow around a circular cylinder is to approximate it as potential flow up to the separation point $\theta=\theta_{s}$, so that for $\theta_{s} \leq \theta \leq \pi$ :

$$
\phi=U_{\infty}\left(r+\frac{R^{2}}{r}\right) \cos \theta
$$

Beyond the separation point ( $0 \leq \theta<\theta_{s}$ ), the cylinder surface pressure is assumed constant and equal to the potential flow value at $\theta=\theta_{s}$. This is shown graphically in following figure, where $C_{p}=\left(p-p_{\infty}\right) /\left(\frac{1}{2} \rho U_{\infty}^{2}\right)$.

![](https://lambda-feedback-prod-frontend-client-bucket.s3.eu-west-2.amazonaws.com/97c443aa-a1ad-494e-9277-54bcaa258dc3/ad69050e-0a87-49d7-a10f-8a746a213388.png){ width=60% }
"""
part_markdown = """
Taking $\theta_{s}=99^{\circ}$ (i.e. separation $81^{\circ}$ from the front stagnation point), calculate the variation of $C_{p}$ on the surface.

When $0 < \theta \leq \theta_s$:
"""
pre_response_text = """
$C_p=$
"""
student_answer = """
1-4 sin( theta)^2
"""
post_response_text = """
"""
correct_answer = """
1-4sin(thetas)^2
"""
print(eval_with_feedback(question_markdown, part_markdown, pre_response_text, student_answer, post_response_text, correct_answer))