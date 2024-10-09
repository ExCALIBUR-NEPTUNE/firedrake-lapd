# Rogers-Ricci

Model based on the finite difference implementation described in "*Low-frequency turbulence in a linear magnetized plasma*", B.N. Rogers and P. Ricci, PRL **104**, 225002, 2010 ([link](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.104.225002)).

Script: [rogers-ricci.py](../scripts/rogers-ricci.py)

## Equations

Rogers & Ricci solve the electrostatic Braginskii equations with $T_i \ll T_e$ and $\beta \ll 1$. Note that there is a factor of $1/m_i$ missing from the equation for ion velocity in the paper.

$$
\begin{aligned}
\frac{d n}{dt} &= -\nabla_{\parallel}(n u_e) + S_n ~~~(1)\\
\frac{d u_i}{dt} &= -u_i\nabla_{\parallel}(u_i) - \frac{1}{m_i n} \nabla_{\parallel}(p_e)~~~(2)\\
m_e\frac{d u_e}{dt} &= -m_e u_e\nabla_{\parallel}(u_e) - \frac{T_e}{n}\nabla_{\parallel}(n) + e\nabla_{\parallel}\phi - 1.71\nabla_{\parallel}T_e + \frac{e j_\parallel}{\sigma_\parallel}~~~(3)\\
\frac{d T_e}{dt} &= \frac{2}{3}\frac{T_e}{e n}0.71\nabla_\parallel j_\parallel - \frac{2}{3}T_e\nabla_{\parallel}(u_e) - u_e\nabla_{\parallel}T_e + S_T~~~(4)\\
\frac{d \omega}{dt} &= -u_i\nabla_\parallel \omega + \frac{m_i \Omega_{ci}^2}{e^2 n}\nabla_\parallel j_\parallel~~~(5)\\
\nabla^2\phi &= \frac{eB^2}{m_i \bar{n}}\omega~~~(6)\\
\end{aligned}
$$

where

$$
\begin{aligned}
u_x &\equiv V_{\parallel x} \\
\frac{df}{dt} &= \frac{\partial f}{\partial t} - \frac{1}{B}\left[\phi,f\right] \\
p_e &= nT_e \\
j_\parallel &= e n (u_i-u_e) \\
\sigma_\parallel &= \frac{e^2 n_0 R}{m_{i}c_{s0}\nu} \\
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

| Parameter      | Value               | Comment                                                                                                                               |
| -------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| $T_0$          | 6 eV                |                                                                                                                                       |
| $L_z$          | 18 m                |                                                                                                                                       |
| $n_0$          | 2e18 m<sup>-3</sup> |                                                                                                                                       |
| $\nu$          | 0.03                |                                                                                                                                       |
| $m_i$          | 6.67e-27 kg         | Inferred from the value of $c_{s0}$ quoted in the paper. Value is $\sim 4 m_p$, consistent with a Helium plasma                       |
| $\tau=m_i/m_e$ | 400                 | Ion/electron mass ratio                                                                                                               |
| $\Omega_{ci}$  | $9.6e5$             |                                                                                                                                       |
| $\Lambda$      | 3                   | Couloumb Logarithm                                                                                                                    |
| R              | 0.5 m               | Approx radius of the plasma column; perpendicular domain size adds a buffer around this to make everything less sensitive to the BCs. |

Derived values
| Parameter          | Calculated as                  | Value                               | Comment                                           |
| ------------------ | ------------------------------ | ----------------------------------- | ------------------------------------------------- |
| B                  | $\Omega_{ci} m_i q_E$          | 40 mT                               |                                                   |
| $c_{s0}$           | $\sqrt{T_0/m_i}$               | 1.2e4 ms<sup>-1</sup>               |                                                   |
| $m_e$              | $\tau m_i$                     | 2.67e-24 kg                         |                                                   |
| $\rho_{s0}$        | $c_{s0}/\Omega{ci}$            | 1.2e-2 m                            | Paper has 1.4e-2 m ... implies $m_i\sim 3 m_p$ !? |
| $S_{0n}$           | 0.03 $n_0 c_{s0}/R$            | 4.8e22 m<sup>-3</sup>s<sup>-1</sup> |                                                   |
| $S_{0T}$           | 0.03 $T_0 c_{s0} / R$          | 4318.4 Ks<sup>-1</sup>              |                                                   |
| $\sigma_\parallel$ | $e^2 n_0 R / (\nu m_i c_{s0})$ | 10676.0                             |                                                   |

### Other implementation details

#### Normalisation

Normalisations follow those in Rogers & Ricci, that is:

|                       | Normalised to |
| --------------------- | ------------- |
| Charge                | $e$           |
| Electric potential    | $e/T_0$       |
| Energy                | $T_0$         |
| Mass<sup>*</sup>      | $m_i$         |
| Number densities      | $n_0$         |
| Perpendicular lengths | $\rho_{s0}$   |
| Parallel lengths      | $R$           |
| Time                  | $R/c_{S0}$    |

<sup>*</sup> Not stated in the paper; but inferred from other normalisation factors.

The normalised forms of the equations are:

$$
\begin{align}
\frac{\partial n}{\partial t}   &= \frac{R}{\rho_{s0}}\left[\phi,n\right] - \nabla_{\parallel}(n u_e) + S_n  ~~({\bf 7}) \\
\frac{\partial u_i}{\partial t} &= \frac{R}{\rho_{s0}}\left[\phi,u_i\right]- u_i\nabla_\parallel u_i - \frac{1}{n}\nabla_\parallel (n T_e) ~~({\bf 8})\\
\frac{\partial u_e}{\partial t} &= \frac{R}{\rho_{s0}}\left[\phi,u_e\right]- u_e\nabla_\parallel u_e - \tau\frac{T_e}{n}\nabla_\parallel n + \tau\nabla_\parallel \phi - 1.71\tau\nabla_\parallel T_e + \tau\nu n(u_i-u_e)  ~~({\bf 9})\\
\frac{\partial T_e}{\partial t} &= \frac{R}{\rho_{s0}}\left[\phi,T_e\right] + \frac{2}{3}\frac{T_e}{n}0.71\nabla_\parallel \left[n (u_i-u_e)\right] - \frac{2}{3}T_e\nabla_\parallel u_e - u_e\nabla_\parallel T_e + S_T ~~({\bf 10})\\
\frac{\partial \omega}{\partial t} &= \frac{R}{\rho_{s0}}\left[\phi,\omega\right] - u_i\nabla_\parallel \omega + \frac{1}{n}\nabla_\parallel \left[n (u_i-u_e)\right] ~~({\bf 11}) \\
\nabla_\perp^2\phi &= \omega ~~({\bf 12}) \\
\end{align}
$$

with 

$$
\begin{equation}
S_n = S_T = 0.03\left\\{1-\tanh[(\rho_{s0}r-r_s)/L_s]\right\\}
\end{equation}
$$

where $R$, $\rho_{s0}$, $r_s$ and $Ls$ are in SI units and $\tau$ and $\nu$ are dimensionless constants, as listed in the tables above.
This system can be be obtained by applying the normalisation factors to equations 1-6, then simplifying; see [here](./details/rogers-ricci-3d-normalised.md) for details. Note that the prime notation used in the derivations is dropped in the equations above for the sake of readability.

#### Boundary conditions
Bohm BCs for velocities at the end walls; zero potential on the transverse boundaries. All other BCs are (homogeneous) Neumann.
That is

| Field    | Transverse BCs | Low-z Parallel BC ($z=-L_z/2$)  | High-z parallel BC ($z=L_z/2$) |
| -------- | -------------- | ------------------------------- | ------------------------------ |
| n        | Neumann        | Neumann                         | Neumann                        |
| T        | Neumann        | Neumann                         | Neumann                        |
| $\omega$ | Neumann        | Neumann                         | Neumann                        |
| $\phi$   | 0              | Neumann                         | Neumann                        |
| $u_i$    | Neumann        | $-c_s$                          | $c_s$                          |
| $u_e$    | Neumann        | $-c_s exp(\Lambda - e\phi/T_e)$ | $c_s exp(\Lambda - e\phi/T_e)$ |

#### Domain and mesh

The mesh is a cuboid with the origin at the centre and dimensions

|                    | Calculated as                  | SI    | Normalised |
| ------------------ | ------------------------------ | ----- | ---------- |
| Parallel size      | $L_z$                          | 18 m  | 36         |
| Perpendicular size | $100\sqrt{T_0/m_i}/\Omega{ci}$ | 1.2 m | 100        |

By default, there are 64x64x16 tetrahedral (cuboidal), giving element sizes of

|                    | SI       | Normalised |
| ------------------ | -------- | ---------- |
| Parallel size      | 1.125 m  | 2.25       |
| Perpendicular size | 1.875 cm | 1.56       |

(Default res is substantially lower than that used in the finite difference model, which has 1024x1024x64 elements.)

#### Simulation time

Rogers and Ricci don't specify the duration of their simulations, except to say that it's sufficient for the system to reach equilibrium.
In a paper by Shi et al. [1], which uses the Gyrokinetic code Gkeyll to model a virtually identical system, they suggest that the simulation should run for "several ion crossing times".
The ion crossing time for our simulations is $\frac{L_z}{2} / \sqrt{T_0/m_i} = 750 \mu{\rm s}$. We set a total duration of 5000 $\mu{\rm s}$

<!-- ## Example output

### CG version

<figure>
  <img src="media/a.gif" width="600">
  <figcaption></figcaption>
</figure>

### DG version

<figure>
  <img src="media/b.gif" width="600">
  <figcaption></figcaption>
</figure> -->

## Weak Forms

### CG version

Equations 1-5 are discretised over a domain $\Omega$ using a continuous Galerkin formulation.
The functions $n$, $u_i$, $u_e$, $T$ and $\omega$, and respective test functions $v_1$, $v_2$, $v_3$, $v_4$ and $v_5$ live in separate CG function spaces ($V_1$ - $V_5$) which need not be of the same polynomial order.

The weak form of the equations, below are written using the shorthand $\left< f(n), v_1 \right> \equiv \int_\Omega f(n) v_1 d\mathbf{x}$ to indicate the standard process of multiplying terms by a test function and integrating over the domain. In practice we look for a solution in the combined function space $V=V_1\times V_2\times V_3\times V_4\times V_5$ where

$$
\begin{aligned}
&\left< \dot{n}, v_1 \right> + \left<\nabla_{\parallel}(n u_e), v_1 \right> - \left< S_n, v_1 \right>\\
&+\left< \dot{u_i}, v_2 \right> + \left< u_i\nabla_{\parallel}(u_i), v_2 \right> + \left< \frac{1}{n} \nabla_{\parallel}(p_e), v_2 \right>\\
&+\left< m_e\dot{u_e}, v_3 \right> + \left< m_e u_e\nabla_{\parallel}(u_e), v_3 \right> + \left< \frac{T_e}{n}\nabla_{\parallel}(n), v_3 \right> - \left< e\nabla_{\parallel}\phi, v_3 \right> + \left< 1.71\nabla_{\parallel}T_e, v_3 \right> - \left< \frac{e j_\parallel}{\sigma_\parallel}, v_3 \right>\\
&+\left< \dot{T_e}, v_4 \right> - \left< \frac{2}{3}\frac{T_e}{e n}0.71\nabla_\parallel j_\parallel, v_4 \right> + \left< \frac{2}{3}T_e\nabla_{\parallel}(u_e), v_4 \right> + \left< u_e\nabla_{\parallel}T_e, v_4 \right> - \left< S_T, v_4 \right>\\
&+\left< \dot{\omega}, v_5 \right> + \left< u_i\nabla_\parallel \omega, v_5 \right> - \left< \frac{m_i \Omega_{ci}^2}{e^2 n}\nabla_\parallel j_\parallel, v_5 \right>\\
&= 0
\end{aligned}
$$

### DG version

WIP



\[1\] [Shi EL, Hammett GW, Stoltzfus-Dueck T, Hakim A. Gyrokinetic continuum simulation of turbulence in a straight open-field-line plasma. Journal of Plasma Physics. 2017;83(3):905830304. doi:10.1017/S002237781700037X](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/gyrokinetic-continuum-simulation-of-turbulence-in-a-straight-openfieldline-plasma/7E4273E83115F9ED55EDA393190D756F)