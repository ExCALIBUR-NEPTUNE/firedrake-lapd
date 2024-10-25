"""
Construct forms for some terms common to multiple scripts.
"""


def poisson_bracket(phi,f):
    return phi.dx(0) * f.dx(1) - phi.dx(1) * f.dx(0)
