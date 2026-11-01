# OD-QC-001 — Ekpyrotic contraction and classicality

**Status:** completed (numerical study; all claims below are reproduced by the
executed notebook unless marked otherwise).

**Artifacts**

- [notebooks/OD-QC-001_ekpyrotic_classicality.ipynb](../notebooks/OD-QC-001_ekpyrotic_classicality.ipynb) — executed notebook (steps 1–5)
- [notebooks/make_notebook.py](../notebooks/make_notebook.py) — notebook generator (regenerate, then `jupyter nbconvert --execute --inplace`)
- [src/ekpyrotic.py](../src/ekpyrotic.py) — model and diagnostics module
- [results/metrics.json](metrics.json) — all quantitative outputs
- Figures: [fig1_smoothing.png](figs/fig1_smoothing.png), [fig2_convergence.png](figs/fig2_convergence.png), [fig3_bounce.png](figs/fig3_bounce.png), [fig4_classicality.png](figs/fig4_classicality.png),
  plus step-6 follow-ups: [fig5_stability.png](figs/fig5_stability.png), [fig6_convergence.png](figs/fig6_convergence.png), [fig7_cascade.png](figs/fig7_cascade.png)

**Model.** Flat FLRW + canonical scalar $\phi$ with exponential potential
$V = s V_0 e^{-c\phi}$, $c=\sqrt{2/p}$, $s=-1$ for $p<1/3$ (ekpyrotic,
$w>1$) and $s=+1$ for $p>1/3$ ($w<1$ counterexample). Reduced Planck units
($3H^2=\rho$). Scaling attractor $a\propto|t|^p$, $w=-1+2/(3p)$. Anisotropy
is an adiabatic shear density $\rho_\Sigma\propto a^{-6}$ (exact for Bianchi I
with isotropic source). Integration in contraction e-folds $n=-\ln a$ (see
numerical notes); $H$ evolved via Raychaudhuri so the Friedmann constraint is
an independent residual. Parameters: $V_0=1$, $t_i=-10^6$, initial shear
fraction $10^{-4}$, GR integration cutoff $\rho_{\rm stop}=10^{10}$ (a trust
cutoff, not a feature of the solution).

---

## Step 1 — Contraction for different equations of state

Fitted power laws vs theory (fit window: while $\rho_\Sigma/\rho_\phi<10^{-3}$;
from `metrics.json`, fits with $R^2 \ge 0.99999999$):

| $p$ | $w$ | $\rho_\phi\propto a^{-3(1+w)}$: fit / theory | $\rho_\Sigma/\rho_\phi\propto a^{3(w-1)}$: fit / theory |
|---|---|---|---|
| 0.01 | 65.67 | $-199.9999$ / $-200$ | $193.9999$ / $194$ |
| 0.05 | 12.33 | $-40.0000$ / $-40$ | $34.0000$ / $34$ |
| 0.10 | 5.67  | $-20.0000$ / $-20$ | $14.0000$ / $14$ |
| 0.30 | 1.22  | $-6.6665$ / $-6.667$ | $0.6665$ / $0.667$ |
| 0.50 | 0.33  | $-4.0007$ / $-4$ | $-1.9993$ / $-2$ |

**Predictions (verified).** For $w>1$ the anisotropy ratio falls as
$a^{3(w-1)}$ during contraction — the ekpyrotic smoothing, reproduced to 4–6
significant digits. The $w=1/3$ counterexample shows the ratio *growing*
during contraction ($a^{-2}$) until the shear leaves the perturbative window:
$w>1$ is the operative condition, not contraction per se. Note the mechanism
needs only $O(1)$ e-folds: the ekpyrotic phase lasts
$n \simeq \tfrac{p}{2}\ln(\rho_f/\rho_i)$ e-folds (e.g. 1.5 e-folds here for
$p=0.1$), and the suppression per e-fold is $e^{-3(w-1)}$, huge for $w\gg1$.

## Step 2 — Sensitivity and convergence

- **Tolerance sweep** (DOP853, rtol $10^{-7}\!\to\!10^{-13}$): the Friedmann
  constraint residual (independent of the evolution: we evolve Raychaudhuri,
  not the constraint) scales with the tolerance,
  $3.2\times10^{-6}\to1.7\times10^{-12}$ relative to $\rho_{\rm tot}$;
  self-convergence of $u(n)$ improves by ~2 orders per decade of rtol; the
  fitted exponent is stable at $13.99998$ (theory 14).
- **Shear amplitude** $10^{-6}, 10^{-4}, 10^{-2}$: exponent unchanged; final
  ratio linear in the initial fraction
  ($3.420\times10^{-23}, 3.420\times10^{-21}, 3.434\times10^{-19}$).
- **$V_0$** $=0.1,1,10$: exponent unchanged (13.999978).

## Step 3 — GR contraction vs a specified effective bounce

The GR solution ends in a crunch at finite time; no bounce exists in the
model. We then specify the effective dynamics $H^2=\tfrac{\rho}{3}(1-\rho/\rho_c)$
with continuity unchanged (hence
$\dot H=-\tfrac12(\rho+p)(1-2\rho/\rho_c)$). Findings:

- The effective model bounces **exactly** at $\rho=\rho_c$
  ($\rho(t_b)=0.041000/0.410000/4.100000$ for the three $\rho_c$ values),
  where the GR solution would reach the same density at essentially the same
  scale factor ($a_{\rm bounce}$ vs $a_{\rm GR}(\rho_c)$: 0.2513/0.2473,
  0.2240/0.2204, 0.1996/0.1964).
- By $\rho=0.41$ ($\approx$ the LQC-motivated critical density) the
  anisotropy ratio has fallen by $6.4\times10^{-10}$ for $p=0.1$ — smoothing
  is complete well before the bounce under these assumptions.
- **Recollapse:** with $V<0$ and no additional physics, the expanding branch
  is not sustained: the kinetic energy redshifts ($\propto a^{-6}$) below
  $|V|$ almost immediately — the post-bounce expansion lasts
  $a_{\max}/a_{\rm bounce}\approx1.03$ — and the model recollapses.
  Thereafter this ad-hoc dynamics shows a cascade of bounces accumulating in
  finite time (visible as the spike region in fig 3c), which we treat as a
  model artifact, not physics. A viable bounce-to-hot-big-bang connection
  needs ingredients this exercise deliberately omits (ghost-condensate /
  galileon modifications of the ekpyrotic phase, reheating, or a potential
  that turns positive).

## Step 4 — WKB classicality

For the minisuperspace WDW equation
$\big[\hbar^2(\partial_\alpha^2-6\partial_\phi^2)+12a^6V\big]\Psi=0$
(arbitrary-ordering-free choice), the WKB classicality quantity
$\mathcal Q=\hbar|\nabla^2 S|/|(\nabla S)^2|$ along the numerical ekpyrotic
trajectory follows the attractor prediction
$\mathcal Q\propto a^{(1-3p)/p}$ to better than 4 significant digits for all
$p$ (including $-1$ for the $w<1$ case: classicality *degrades* as the shear
takes over — a consistent picture). For $p<1/3$, $\mathcal Q\to0$ as the
crunch is approached: **the semiclassical description of the contracting
branch improves monotonically through the ekpyrotic phase** — the local
statement behind the classicality results in the ekpyrotic wavefunction
literature (Battarra & Lehners 2014; Lehners 2015). With $\hbar=1$ and these
units $\mathcal Q(t_i)\sim5\times10^{6}$ and falls below 1 at
$a/a_i\approx0.11$ for $p=0.1$; its overall normalization scales with
$\hbar$, so only the exponent is convention-independent.

## What is assumed (not derived)

1. Flat slicing, single canonical scalar, exact exponential potential.
2. Shear as an adiabatic stiff component (exact for homogeneous Bianchi I;
   inhomogeneous anisotropies are not treated).
3. The effective Friedmann equation of step 3 — specified by hand, in the
   style of LQC effective dynamics; no quantum derivation here.
4. WDW factor ordering (first-derivative-free); $\hbar$ as a free overall
   factor in $\mathcal Q$; minisuperspace truncation (perturbative modes,
   which carry the observables, are absent).
5. Initial data on the scaling attractor (attractivity itself is known
   analytically and was spot-checked but not re-proven here).

## What the models predict

1. Smoothing exponent $3(w-1)$ exactly; suppression linear in initial shear
   fraction; insensitive to $V_0$.
2. No GR bounce; an ad-hoc effective dynamics bounces cleanly at
   $\rho=\rho_c$ but recollapses immediately for $V<0$.
3. WKB classicality exponent $(1-3p)/p$: the contracting branch becomes
   arbitrarily classical in the WKB sense as $\rho$ grows, for $w>1$.

## What remains unverified / open

1. The post-bounce phase: matching onto a hot expanding universe (requires
   the extra physics listed in step 3).
2. Perturbations across the bounce and the quantum boundary proposal
   (no-boundary vs tunneling): our step 4 is a WKB self-consistency check
   along one classical history, not a statement about which history the
   wavefunction of the universe selects — the content Lehners' 2015 analysis
   adds via instanton/saddle methods.
3. The bounce-accumulation cascade seen in the effective model: is it a
   generic pathology of $H^2=\frac{\rho}{3}(1-\rho/\rho_c)$ with negative
   potentials, or an artifact of our simplified closure? Worth a follow-up.
   *(Now characterized quantitatively — see Follow-ups: Zeno accumulation,
   $\Delta t_n\propto n^{-2}$, $|u|_n\propto n$.)*
4. Inhomogeneous anisotropy gradients (omitted from Bianchi I), expected to
   grow as $a^{-2}$ — verified in the literature, not here.
5. The marginal (time-translation) direction of the ekpyrotic attractor
   quantified below is a numerical-experiment artifact of homogeneous
   perturbations; whether generic perturbations of homogeneous-but-anisotropic
   initial data converge to the attractor as cleanly deserves a dedicated
   sweep (varying initial $u$, $H$ off-attractor).
   *(The pure-scalar case is now settled — see Follow-ups: decay at
   $3(w-1)/2$ plus a time shift; the anisotropic sector remains open.)*

## Numerical notes (method reliability)

- **e-fold time.** In cosmic time the integrator drifts onto a time-shifted
  attractor near the crunch; in $n=-\ln a$ the attractor is exactly
  exponential and the same accuracy holds uniformly to the cutoff. All
  physical results are quoted in $t$-invariant variables ($a$, $n$, $\rho$).
- **Marginal time-translation mode.** The ekpyrotic scaling solution has an
  exact time-translation symmetry; numerically, any homogeneous perturbation
  (including the shear backreaction at fraction $f$) displaces the measured
  crunch time by $\delta t_c \propto f$ (measured: exact linearity over
  $f=10^{-6}\dots10^{-3}$; $\delta t_c/|t_i|\approx -36.8\,f$ for $p=0.1$).
  This is a coordinate effect and does not contaminate $t$-invariant
  observables; it does mean absolute times must be quoted relative to a
  measured $t_c$, as done in fig 3.
- **Constraint monitoring.** $|\mathcal C|/\rho_{\rm tot}\lesssim10^{-10}$
  (rtol $=10^{-11}$) along all runs; the effective model's modified
  constraint is monitored analogously
  ($|{\mathcal C}_{\rm eff}|\sim10^{-9}$).
- Fits are log-log least squares, restricted to the perturbative window
  $\rho_\Sigma/\rho_\phi<10^{-3}$; all reported $R^2\ge0.999999$ except
  where noted.

## Follow-ups (run 2): attractor stability, the bounce cascade, a failed remedy

*(Step 6 of the notebook; eigenvalues, cascade law, and breakdown times are
in `metrics.json` under `steps['6a']`–`steps['6c']`.)*

1. **Attractor stability — the $w=1$ threshold is the exact linear
   stability boundary.** In the $t$-invariant scaled coordinates
   $X=-c(\phi-\phi_{\rm att})$, $U=u/u_{\rm att}$, $H_h=h/h_{\rm att}$ the
   pure-scalar flow reduces exactly to the autonomous system
   $\dot X=\frac{2}{p}(U/H_h-1)$,
   $\dot U=(3-\frac1p)U+\frac{1-3p}{p}e^X/H_h$,
   $\dot H_h=(U^2/H_h-H_h)/p$ with fixed point $(0,1,1)$. The linearisation
   has the exact spectrum
   $\lambda=\{0,\;-(1-3p)/p,\;-2/p\}=\{0,\,-\tfrac32(w-1),\,-3(w+1)\}$
   (numeric Jacobian matches to 8 digits). The slow mode decays as
   $a^{3(w-1)/2}$ — exactly half the smoothing exponent — and was measured
   at $6.99$–$7.00$ per e-fold (theory $7$) for $p=0.1$ across $X$, $U$,
   $H_h$. The zero mode is the time shift of the marginal-mode analysis
   above. Since $\lambda_1=0\iff w=1$ and $\lambda_1>0$ for $w<1$, the
   scaling solution is linearly **unstable** precisely in the $w<1$ regime
   where it also fails to smooth — one condition, two roles.

2. **The bounce cascade is a Zeno accumulation.** Continuing the effective
   model past its first recollapse: 116 H-crossings resolved down to
   $\rho=10^{-8}\rho_c$ (115 cycles), cycle durations following
   $\Delta t_n = 0.09/n^{2.01}$ (max relative residual $1.5\times10^{-2}$),
   field velocity at the crossings growing as $|u|_n = 5.19\,n+2.2$, and
   $\phi$ gliding from $-0.21$ toward $\approx-2.8$. The $\sum 1/n^2$
   series accumulates at $t_{\rm acc}\approx-3.6254$: **infinitely many
   bounces with unbounded kinetic energy in finite time** — a Zeno-type
   breakdown of the effective closure, not physics. At each recollapse the
   state is degenerate: $\rho\to0$ requires $u^2/2=|V|$ to ~10 digits
   (at the first recollapse, $u\approx-711$ against $|V|\approx2.5\times10^5$).

3. **Flipping the sign of the potential does not rescue the expansion.**
   With $V>0$ in the expanding branch (crude stand-in for "the potential
   turns positive"), the post-bounce velocity $u_b=-2.44$ points toward
   larger $|V|$; the on-shell expansion is driven back across the
   constraint boundary $\rho=\rho_c$ within $\Delta t\lesssim0.2$, where the
   integration breaks down. Together with (2): a working ekpyrotic bounce
   must change the field's post-bounce roll (as ghost-condensate and
   galileon mechanisms do), not merely deform $H^2$.

## References (context; not reproduced here)

- Khoury, Ovrut, Steinhardt, Turok, *The Ekpyrotic Universe: Colliding
  Branes and the Origin of the Hot Big Bang*, Phys. Rev. D 64 (2001).
- Buchbinder, Khoury, Ovrut, *New Ekpyrotic Cosmology*, Phys. Rev. D 76
  (2007) — bounce via ghost condensation.
- Battarra, Lehners, *On the No-Boundary Proposal for Ekpyrotic and Cyclic
  Cosmologies*, JCAP 12 (2014).
- Lehners, *Classical inflationary and ekpyrotic universes in the
  no-boundary wavefunction*, Phys. Rev. D 91 (2015) — the "Lehners 2015"
  classicality analysis motivating step 4.
- Ashtekar, Pawlowski, Singh, *Quantum nature of the big bang*, Phys. Rev.
  Lett. 96 (2006) — origin of the effective Friedmann equation used in
  step 3.
