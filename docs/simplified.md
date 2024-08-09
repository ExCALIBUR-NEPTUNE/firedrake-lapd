# Simplified system

## Equations

$$
\dot{n} + \nabla \cdot \left ( n \left ( v_{\parallel} + v_D \right )\right ) = n_0
$$

$$
\dot{(n v_{\parallel})} + \nabla \cdot \left ( n v_{\parallel} \left ( v_{\parallel} + v_D \right )\right ) = - T \nabla_{\parallel} n
$$

$$
\dot{\omega} + \nabla \cdot \left ( \omega \left ( v_{\parallel} + v_D \right )\right ) = - \nabla_{\parallel} (n v_{\parallel})
$$

$$
\nabla_{\perp}^2 \phi = \omega
$$

$$
v_D = \left ( \frac{\partial \phi}{\partial y}, -\frac{\partial \phi}{\partial x}, 0 \right )
$$

is the gradient RHS for omega needed?

## Reduction to form used in Firedrake

The first two eqs are

$$
\dot{n} + \nabla \cdot \left ( n \left ( v_{\parallel} + v_D \right )\right ) = n_0
$$

$$
\dot{(n v_{\parallel})} + \nabla \cdot \left ( n v_{\parallel} \left ( v_{\parallel} + v_D \right )\right ) = - T \nabla_{\parallel} n.
$$

By writing $\dot{(n v_{\parallel})} \equiv n \dot{v_{\parallel}} + \dot{n} v_{\parallel}$ and substituting from the first equation one obtains for $v_{\parallel}$

$$
n \dot{v_{\parallel}} + v n_0 + n \nabla{v_{\parallel}} \cdot (v_{\parallel}+v_D) = -T \nabla_{\parallel} n.
$$

(If unsure about the vector calculus see the [vector calculus identities Wiki page](https://en.wikipedia.org/wiki/Vector_calculus_identities), that's what I do.)

## CG version

Script: [(LAPD-like_simplified_CG.py)](../scripts/LAPD-like_simplified_CG.py)

Implements the equations above, plus an attempt at streamline-upwind correction to add artificial viscosity.

<p float="left">
  <img src="./media/CGreduced_anim_density_64by64.gif" width="400">
  <img src="./media/CGreduced_anim_density_midslice_64by64.gif" width="400">
</p>

## DG version

Work in progress

<!-- Script: [(LAPD-like_simplified_DG.py)](../scripts/LAPD-like_simplified_DG.py) -->