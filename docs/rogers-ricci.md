# Rogers-Ricci

Model based on the finite difference implementation described in "*Low-frequency turbulence in a linear magnetized plasma*", B.N. Rogers and P. Ricci, PRL **104**, 225002, 2010 ([link](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.104.225002)).

## Equations

(Electrostatic Braginskii equations with $T_i \ll T_e$ and $\beta \ll 1$)

$$
\begin{aligned}
\dot{n} &= -\nabla_{\parallel}(n u_e) + S_n ~~~(1)\\
\dot{u_i} &= -u_i\nabla_{\parallel}(u_i) - \frac{1}{n} \nabla_{\parallel}(p_e)~~~(2)\\
m_e\dot{u_e} &= -m_e u_e\nabla_{\parallel}(u_e) - \frac{T_e}{n}\nabla_{\parallel}(n) + e\nabla_{\parallel}(\phi) - 1.71\nabla_{\parallel}(T_e) + \frac{e j_\parallel}{\sigma_\parallel}~~~(3)\\
\dot{T_e} &= \frac{2}{3}\frac{T_e}{e n}0.71\nabla_\parallel j_\parallel - \frac{2}{3}T_e\nabla_{\parallel}(u_e) - u_e\nabla_{\parallel}(T_e) + S_T~~~(4)\\
\dot{\omega} &= -u_i\nabla_\parallel \omega + \frac{m_i \Omega_{ci}^2}{e^2 n}\nabla_\parallel j_\parallel~~~(5)\\
\nabla_{\perp}^2 \phi &= \omega
\end{aligned}
$$

where

$$
\begin{aligned}
u_x &\equiv V_{\parallel x} \\
p_e &= nT_e \\
j_\parallel &= e n (u_i-u_e) \\
\sigma_\parallel &= \frac{e^2 n_0 R}{m_{i}c_{s0}/\nu} \\
\Omega_{ci} &= e B / m_i c
\end{aligned}
$$

and the source terms have the form

$$
\begin{aligned}
S_n &= S_{0n}\frac{1-{\rm tanh[(r-r_s)/L_s]}}{2} \\
S_T &= S_{0T}\frac{1-{\rm tanh[(r-r_s)/L_s]}}{2} \\
\end{aligned}
$$

where $r = \sqrt{x^2 + y^2}$

## Parameter Choices

| Parameter     | Value               | Comment                                                                                                         |
| ------------- | ------------------- | --------------------------------------------------------------------------------------------------------------- |
| $T_{e0}$      | 6 eV                |                                                                                                                 |
| $L_z$         | 18 m                |                                                                                                                 |
| $n_0$         | 2e18 m<sup>-3</sup> |                                                                                                                 |
| $\nu$         | 0.03                |                                                                                                                 |
| $m_i$         | 6.67e-27 kg         | Inferred from the value of $c_{s0}$ quoted in the paper. Value is $\sim 4 m_p$, consistent with a Helium plasma |
| $\Omega_{ci}$ | $9.6e5$             |                                                                                                                 |
| $\Lambda$     | 3                   | Couloumb Logarithm                                                                                              |
| R             | 0.5 m               | Approx radius of the plasma column?                                                                             |

Derived values
| Parameter          | Calculated as                   | Value                               | Comment                                           |
| ------------------ | ------------------------------- | ----------------------------------- | ------------------------------------------------- |
| B                  | $\Omega_{ci} m_i q_E$           | 40 mT                               |                                                   |
| $c_{s0}$           | $\sqrt{T_{e0}/m_i}$             | 1.2e4 ms<sup>-1</sup>               |                                                   |
| $\rho_{s0}$        | $c_{s0}/\Omega{ci}$             | 1.2e-2 m                            | Paper has 1.4e-2 m ... implies $m_i\sim 3 m_p$ !? |
| $S_{0n}$           | 0.03 $n_0 c_{s0}/R$             | 4.8e22 m<sup>-3</sup>s<sup>-1</sup> |                                                   |
| $S_{0T}$           | 0.03 $T_{e0} c_{s0} / R$        | 4318.4 Ks<sup>-1</sup>              |                                                   |
| $\sigma_\parallel$ | $e^2 n_0 R / (0.03 m_i c_{s0})$ | 10676.0                             |                                                   |

### Other implementation details

#### Boundary conditions
Bohm BCs for velocities at the end walls ($z = \pm L_z/2$): $u_i= \pm c_s$, $u_e=\pm exp(\Lambda - e\phi/T_e)$. All other BCs are homogeneous Neumann.


#### Domain and mesh

The mesh is a cuboid with the origin at the centre and dimensions

|                    | Determined by params           | SI    | Normalised |
| ------------------ | ------------------------------ | ----- | ---------- |
| Parallel size      | $L_z$                          | 18 m  | 36         |
| Perpendicular size | $\sqrt{T_{e0}/m_i}/\Omega{ci}$ | 1.2 m | 100        |

By default, there are 64x64x16 tetrahedral (cuboidal), giving element sizes of

|                    |          | Normalised |
| ------------------ | -------- | ---------- |
| Parallel size      | 1.125 m  | 2.25       |
| Perpendicular size | 1.875 cm | 1.56       |

(Default res is substantially lower than that used in the finite difference model, which has 1024x1024x64 elements.)

#### Normalisation

Normalisations follow those in Rogers & Ricci, that is:

|                       | Normalised to   |
| --------------------- | --------------- |
| Charge                | $e$             |
| Electric potential    | $e/T_{e0}$      |
| Energy                | $T_{e0}$        |
| Mass<sup>*</sup>      | $1600 m_i$      |
| Number densities      | $n_0$           |
| Perpendicular lengths | $100 \rho_{s0}$ |
| Parallel lengths      | $R$             |
| Time                  | $R/c_{S0}$      |

<sup>*</sup> Not stated in the paper; derived from other normalisation factors.

#### Simulation time

Based on Fig 4. of Rogers & Ricci, looks like total simulation time is $\sim 12$.
Assuming this, like other figures, is in normalised units, we need $t_{\rm end}=12 R/c_{s0} = 500 \mu{\rm s}$ before normalisation.

## CG version

Script: [rogers-ricci_CG.py](../scripts/rogers-ricci_CG.py)

### Weak Form

Equations 1-5 are discretised over a domain $\Omega$ using a continuous Galerkin formulation.
The functions $n$, $u_i$, $u_e$, $T$ and $\omega$, and respective test functions $v_1$, $v_2$, $v_3$, $v_4$ and $v_5$ live in separate CG function spaces ($V_1$ - $V_5$) which need not be of the same polynomial order.

The weak form of the equations, below are written using the shorthand $\left< f(n), v_1 \right> \equiv \int_\Omega f(n) v_1 d\mathbf{x}$ to indicate the standard process of multiplying terms by a test function and integrating over the domain. In practice we look for a solution in the combined function space $V=V_1\times V_2\times V_3\times V_4\times V_5$ where

$$
\begin{aligned}
&\left< \dot{n}, v_1 \right> + \left<\nabla_{\parallel}(n u_e), v_1 \right> - \left< S_n, v_1 \right>\\
&+\left< \dot{u_i}, v_2 \right> + \left< u_i\nabla_{\parallel}(u_i), v_2 \right> + \left< \frac{1}{n} \nabla_{\parallel}(p_e), v_2 \right>\\
&+\left< m_e\dot{u_e}, v_3 \right> + \left< m_e u_e\nabla_{\parallel}(u_e), v_3 \right> + \left< \frac{T_e}{n}\nabla_{\parallel}(n), v_3 \right> - \left< e\nabla_{\parallel}(\phi), v_3 \right> + \left< 1.71\nabla_{\parallel}(T_e), v_3 \right> - \left< \frac{e j_\parallel}{\sigma_\parallel}, v_3 \right>\\
&+\left< \dot{T_e}, v_4 \right> - \left< \frac{2}{3}\frac{T_e}{e n}0.71\nabla_\parallel j_\parallel, v_4 \right> + \left< \frac{2}{3}T_e\nabla_{\parallel}(u_e), v_4 \right> + \left< u_e\nabla_{\parallel}(T_e), v_4 \right> - \left< S_T, v_4 \right>\\
&+\left< \dot{\omega}, v_5 \right> + \left< u_i\nabla_\parallel \omega, v_5 \right> - \left< \frac{m_i \Omega_{ci}^2}{e^2 n}\nabla_\parallel j_\parallel, v_5 \right>\\
&= 0
\end{aligned}
$$

<!-- ## DG version

Script: [rogers-ricci_DG.py](../scripts/rogers-ricci_DG.py) -->