## Normalisations

(Normalised quantities indicated by $'$)

$$
\begin{align}
n                &= n_0 n'                                   \\
u_x              &= c_s u_x'                                 \\
\omega           &= n_0\omega'                               \\
T_e              &= T_0 T_e'                                 \\
\phi             &= \frac{T_0}{e}\phi'                       \\
t                &= \frac{R}{c_{s}}t'                        \\
\nabla_\perp     &= \frac{1}{\rho_{s0}}\nabla_\perp'         \\
\nabla_\parallel &= \frac{1}{R}\nabla_\parallel'             \\
[a,b]            &= \left(\frac{1}{\rho_{s0}}\right)^2[a,b]' \\
j_\parallel      &= e n_0 c_s n' (u_i'-u_e')                 \\
L_\perp          &= \rho_{s0}~L_\perp'                       \\
L_\parallel      &= R~L_\parallel'                           \\
\end{align}
$$

## Normalised equations

The derivations below involve starting from the SI version of each equation, substituting in the normalised variables, then reducing the result to as simple a form as possible.
In the first subsection, we demonstrate how to normalise the Poisson bracket terms, which feature in all of the time evolution equations.

### Poisson Brackets

SI: $\frac{df}{dt} = \frac{\partial f}{\partial t} - \frac{1}{B}\left[\phi,f\right]$

---
$$
\begin{align}
\frac{df'}{dt'}\frac{c_s}{R} &= \frac{\partial f'}{\partial t'}\frac{c_s}{R} - \frac{1}{B}\frac{T_0}{e}\left(\frac{1}{\rho_{s0}}\right)^2\left[\phi',f'\right]' \\
\frac{df'}{dt'} &= \frac{\partial f'}{\partial t'} - \frac{T_0 R}{B \rho_{s0}^2 e c_s}\left[\phi',f'\right]' \\
\frac{df'}{dt'} &= \frac{\partial f'}{\partial t'} - \frac{R}{\rho_{s0}}\left[\phi',f'\right]' ~~({\bf 0})\\
\frac{df'}{dt'} &= \frac{\partial f'}{\partial t'} - 40\left[\phi',f'\right]' \\
\end{align}
$$

Where
$\rho_{s0}=\frac{c_s}{\Omega_{ci}}$, $c_s^2=\frac{T_0}{m_i}$ and $\Omega_{ci}=\frac{eB}{m_i}$ ($=> \frac{T_0}{B\rho_{s0}e c_s}=1$) were used to obtain the penultimate line and all factors required to transform $f\mapsto f'$ on the first line cancel.


### Density

SI equation: $\frac{d n}{dt} = -\nabla_{\parallel}(n u_e) + S_n$

---
Let

$$
\begin{align}
S_n' &= S_n \frac{R}{n_0 c_s}\\
&= 0.03\left\\{1-\tanh[(r-r_s)/L_s]\right\\} \\
&= 0.03\left\\{1-\tanh[(\rho_{s0}r'-r_s)/L_s]\right\\} \\
\end{align}
$$

$$
\begin{align}
n_0\frac{c_s}{R}\frac{d n'}{dt'} &= -\frac{c_s}{R}\nabla_{\parallel}'(n' u_e') n_0 + S_n  \\
                \frac{d n'}{dt'} &= -\nabla_{\parallel}'(n' u_e') + S_n\frac{R}{n_0 c_s} \\
                \frac{d n'}{dt'} &= -\nabla_{\parallel}'(n' u_e') + S_n' \\
 \frac{\partial n'}{\partial t'} &= \frac{R}{\rho_{s0}}\left[\phi',n'\right]' - \nabla_{\parallel}'(n' u_e') + S_n' ~~({\bf 1}) \\
 \frac{\partial n'}{\partial t'} &- 40\left[\phi',n'\right]' + \nabla_{\parallel}'(n' u_e') - 0.03\left\\{1-\tanh[(\rho_{s0}r' + r_s)/L_s]\right\\} = 0 \\
\end{align}
$$

### Ion parallel velocity

SI equation: $\frac{d u_i}{dt} = -u_i\nabla_{\parallel}(u_i) - \frac{1}{m_i n} \nabla_{\parallel}(p_e)$

---

$$
\begin{align}
c_s\frac{c_s}{R} \frac{d u_i'}{dt'} &= -\frac{c_s^2}{R}u_i'\nabla_\parallel'u_i' - \frac{1}{m_i n_0 n'}\frac{1}{R}\nabla_\parallel'(n_0 n' T_0 T_e') \\
  \frac{c_s^2}{R}\frac{d u_i'}{dt'} &= -\frac{c_s^2}{R}u_i'\nabla_\parallel'u_i' - \frac{T_0}{m_i R n'}\nabla_\parallel'(n' T_e') \\
                 \frac{d u_i'}{dt'} &= -u_i'\nabla_\parallel'u_i' - \frac{1}{n'}\nabla_\parallel'(n' T_e') \\
  \frac{\partial u_i'}{\partial t'} &= \frac{R}{\rho_{s0}}\left[\phi',u_i'\right]'-u_i'\nabla_\parallel'u_i' - \frac{1}{n'}\nabla_\parallel'(n' T_e') ~~({\bf 2})\\
  \frac{\partial u_i'}{\partial t'} &- 40\left[\phi',u_i'\right]' + u_i'\nabla_\parallel'u_i' + \frac{1}{n'}\nabla_\parallel'(n' T_e') = 0\\
\end{align}
$$

Where $c_s^2=\frac{T_0}{m_i}$ was used to obtain the third line.

### Electron parallel velocity

SI equation: $m_e\frac{d u_e}{dt} = -m_e u_e\nabla_{\parallel}(u_e) - \frac{T_e}{n}\nabla_{\parallel}(n) + e\nabla_{\parallel}(\phi) - 1.71\nabla_{\parallel}T_e + \frac{e j_\parallel}{\sigma_\parallel}$

---

$$
\begin{align}
m_e\frac{d u_e'}{dt'}c_s\frac{c_s}{R} &= -m_e c_s u_e'\frac{c_s}{R}\nabla_\parallel'u_e' - \frac{T_0 T_e'}{n_0 n'}\frac{1}{R}\nabla_\parallel'(n_0 n') + e\frac{1}{R}\nabla_\parallel'(\frac{T_0}{e}\phi') - 1.71\frac{1}{R}\nabla_\parallel'(T_0 T_e') + \frac{e^2 n_0 c_s n' (u_i'-u_e')}{\frac{e^2 n_0 R}{m_{i}c_s\nu}} \\
m_e\frac{d u_e'}{dt'}\frac{c_s^2}{R} &= -m_e u_e'\frac{c_s^2}{R}\nabla_\parallel'u_e' - \frac{T_0}{R}\frac{T_e'}{n'}\nabla_\parallel'n' + \frac{T_0}{R}\nabla_\parallel'(\phi') - 1.71\frac{T_0}{R}\nabla_\parallel'T_e' + \frac{c_s^2}{R} m_{i}\nu n'(u_i'-u_e') \\
m_e\frac{d u_e'}{dt'}\frac{c_s^2}{R} &= -m_e u_e'\frac{c_s^2}{R}\nabla_\parallel'u_e' - m_i\frac{c_s^2}{R}\frac{T_e'}{n'}\nabla_\parallel'n' + m_i\frac{c_s^2}{R}\nabla_\parallel'(\phi') - 1.71m_i\frac{c_s^2}{R}\nabla_\parallel'T_e' + \frac{c_s^2}{R} m_{i}\nu n'(u_i'-u_e') \\
\frac{d u_e'}{dt'} &= -u_e'\nabla_\parallel'u_e' - \tau\frac{T_e'}{n'}\nabla_\parallel'n' + \tau\nabla_\parallel'(\phi') - 1.71\tau\nabla_\parallel'T_e' + \tau\nu n'(u_i'-u_e') \\
\frac{\partial u_e'}{\partial t'} &= \frac{R}{\rho_{s0}}\left[\phi',u_e'\right]'-u_e'\nabla_\parallel'u_e' - \tau\frac{T_e'}{n'}\nabla_\parallel'n' + \tau\nabla_\parallel'(\phi') - 1.71\tau\nabla_\parallel'T_e' + \tau\nu n'(u_i'-u_e')~~({\bf 3})\\
\frac{\partial u_e'}{\partial t'} &- 40\left[\phi',u_e'\right]' + u_e'\nabla_\parallel'u_e' + 400\frac{T_e'}{n'}\nabla_\parallel'n' - 400\nabla_\parallel'(\phi') + 684\nabla_\parallel'T_e' - 12 n'(u_i'-u_e') = 0 \\
\end{align}
$$

Where $T_0 = m_i c_s^2$ was used in obtaining the third line and we have defined $\tau=m_i/m_e$.

### Temperature

SI equation: $\frac{d T_e}{dt} = \frac{2}{3}\frac{T_e}{e n}0.71\nabla_\parallel j_\parallel - \frac{2}{3}T_e\nabla_{\parallel}(u_e) - u_e\nabla_{\parallel}T_e + S_T$

---
Let

$$
\begin{align}
S_T' &= S_T \frac{R}{T_0 c_s}\\
&= 0.03\left\\{1-\tanh[(r-r_s)/L_s]\right\\} \\
&= 0.03\left\\{1-\tanh[(\rho_{s0}r'-r_s)/L_s]\right\\} \\
\end{align}
$$

$$
\begin{align}
\frac{d T_e'}{dt'}\frac{T_0 c_s}{R} &= \frac{2}{3}\frac{T_e'}{e n'}\frac{T_0}{n_0}0.71\frac{1}{R}\nabla_\parallel'\left[e n_0 c_s n' (u_i'-u_e')\right] - \frac{2}{3}\frac{c_s T_0}{R}T_e'\nabla_\parallel'u_e' - \frac{c_s T_0}{R}u_e'\nabla_\parallel'T_e' + \frac{T_0 c_s}{R}S_T' \\
\frac{d T_e'}{dt'}\frac{T_0 c_s}{R} &= \frac{2}{3}\frac{T_e'}{n'}\frac{c_s T_0}{R}0.71\nabla_\parallel'\left[n' (u_i'-u_e')\right] - \frac{2}{3}\frac{c_s T_0}{R}T_e'\nabla_\parallel'u_e' - \frac{c_s T_0}{R}u_e'\nabla_\parallel'T_e' + \frac{T_0 c_s}{R}S_T' \\
\frac{d T_e'}{dt'} &= \frac{2}{3}\frac{T_e'}{n'}0.71\nabla_\parallel'\left[n' (u_i'-u_e')\right] - \frac{2}{3}T_e'\nabla_\parallel'u_e' - u_e'\nabla_\parallel'T_e' + S_T'\\
\frac{\partial T_e'}{\partial t'} &= \frac{R}{\rho_{s0}}\left[\phi',T_e'\right]' + \frac{2}{3}\frac{T_e'}{n'}0.71\nabla_\parallel'\left[n' (u_i'-u_e')\right] - \frac{2}{3}T_e'\nabla_\parallel'u_e' - u_e'\nabla_\parallel'T_e' + S_T' ~~({\bf 4})\\
\frac{\partial T_e'}{\partial t'} &- 40\left[\phi',T_e'\right]' - \frac{71}{150}\frac{T_e'}{n'}\nabla_\parallel'\left[n' (u_i'-u_e')\right] + \frac{2}{3}T_e'\nabla_\parallel'u_e' + u_e'\nabla_\parallel'T_e' - 0.03\left\\{1-\tanh[(\rho_{s0}r'-r_s)/L_s]\right\\} = 0 \\
\end{align}
$$


### Vorticity

SI equation: $\frac{d \nabla_\perp^2\phi}{dt} = -u_i\nabla_\parallel\left(\nabla_\perp^2\phi\right) + \frac{m_i \Omega_{ci}^2}{e^2 n}\nabla_\parallel j_\parallel$

---

$$
\begin{align}
\left(\frac{1}{\rho_{s0}}\right)^2\frac{T_0}{e}\frac{c_s}{R}\frac{d \nabla_\perp'^2\phi'}{dt'} &= -\frac{c_s}{R}\left(\frac{1}{\rho_{s0}}\right)^2\frac{T_0}{e}u_i'\nabla_\parallel'\left(\nabla_\perp'^2\phi'\right) + \frac{m_i \Omega_{ci}^2}{e^2 n_0 n' R}\nabla_\parallel'\left[e n_0 c_s n' (u_i'-u_e')\right] \\
\frac{d \nabla_\perp'^2\phi'}{dt'} &= -u_i'\nabla_\parallel'\left(\nabla_\perp'^2\phi'\right) + \frac{\rho_{s0}^2 e R}{c_s T_0}\frac{m_i \Omega_{ci}^2}{e^2 n_0 n' R}\nabla_\parallel'\left[e n_0 c_s n' (u_i'-u_e')\right] \\
\frac{d \nabla_\perp'^2\phi'}{dt'} &= -u_i'\nabla_\parallel'\left(\nabla_\perp'^2\phi'\right) + \frac{\rho_{s0}^2}{T_0}\frac{m_i \Omega_{ci}^2}{n'}\nabla_\parallel'\left[n' (u_i'-u_e')\right] \\
\frac{d \omega'}{dt'} &= -u_i'\nabla_\parallel'\omega' + \frac{1}{n'}\nabla_\parallel'\left[n' (u_i'-u_e')\right] \\
\frac{\partial \omega'}{\partial t'} &= \frac{R}{\rho_{s0}}\left[\phi',\omega'\right]'-u_i'\nabla_\parallel'\omega' + \frac{1}{n'}\nabla_\parallel'\left[n' (u_i'-u_e')\right] ~~({\bf 5}) \\
\frac{\partial \omega'}{\partial t'} &- 40\left[\phi',\omega'\right]' + u_i'\nabla_\parallel'\omega' - \frac{1}{n'}\nabla_\parallel'\left[n' (u_i'-u_e')\right] = 0 \\
\end{align}
$$

Where  $c_s^2=T_0/m_i=\rho_{s0}^2\Omega_{ci}^2$ was used to obtain the fourth line.

### Potential

SI equation: $\nabla_\perp^2\phi = \frac{eB^2}{m_i \bar{n}}\omega$

---

$$
\begin{align}
\nabla_\perp'^2\phi'\left(\frac{1}{\rho_{s0}}\right)^2\frac{T_0}{e} &= \frac{eB^2}{m_i \bar{n}}n_0\omega' \\
\nabla_\perp'^2\phi'\left(\frac{1}{\rho_{s0}}\right)^2\frac{T_0}{e} &= \frac{\Omega_{ci}^2 m_i}{e\bar{n}}n_0\omega \\
\nabla_\perp'^2\phi' &= \frac{\rho_{s0}^2\Omega_{ci}^2 m_i}{T_0\bar{n}}n_0\omega \\
\nabla_\perp'^2\phi' &= \frac{n_0}{\bar{n}}\omega \\
\nabla_\perp'^2\phi' &= \omega ~~({\bf 6}) \\
\end{align}
$$

Where  $c_s^2=T_0/m_i=\rho_{s0}^2\Omega_{ci}^2$ was used to obtain the penultimate line and the last line follows from the choice $\bar{n}=n_0$.


