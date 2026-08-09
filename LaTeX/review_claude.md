# Review of `article.tex` — "Neutrino masses and mixing angles in the Plan B Model"

*Line numbers refer to `article.tex` as of 8 July 2026. No original files have been modified.*

---

## 1. Technical and scientific issues

Ordered roughly by importance. Items T1–T4 were verified numerically (numpy, exact diagonalisation vs the paper's formulae); the scripts are trivially reproducible on request.

### T1. Normal-hierarchy section: the flavour alignment of the heaviest state contradicts the displayed PMNS matrix (§2.3, lines 610–691)

With the texture $y_{22}=0$, the leading matrix $X^{(0)}$ has its only $\mathcal{O}(1)$ entry in the (1,1) [electron] position (line 633). The heaviest eigenstate is therefore generically aligned with $\nu_e$. If PMNS columns are ordered $m_1<m_2<m_3$ (as they must be to compare with NuFIT NO data), this gives $|U_{e3}|\approx 1$, i.e. $\theta_{13}\approx 90^\circ$ — grossly excluded. Yet the displayed $U$ (lines 676–683) has $U_{e1}\approx 1$ and $U_{e3}\sim\epsilon$, i.e. the *lightest* state e-aligned, which is what the data want but is inconsistent with the eigenvalue analysis in the same subsection ($m_3 = m(A_3+\ldots)$ being the $\mathcal{O}(1)$ eigenvalue whose eigenvector is the e-direction).

Numerical check: drawing $y_{ij},a_{ij}$ as random $\mathcal{O}(1)$ numbers with $y_{22}=0$, $\epsilon=0.42$, the heaviest state is e-aligned (or occasionally τ-aligned) and $\theta_{13}$ comes out anywhere between $\sim15^\circ$ and $\sim80^\circ$ — never generically small. The mass *ratios* work beautifully ($m_2/m_3\sim\epsilon^2$ ✓), but the angles do not follow from $\mathcal{O}(1)$ coefficients alone. The claim at line 691 ("if the $u^{(k)}_{ij}$ parameters are of order 1 and $\epsilon\sim0.4$, then we can fit these approximations to the data") therefore overstates the analytic case. Presumably the nested-sampling posterior lives in regions with additional accidental suppressions (e.g. of $y_{11}^2\alpha_{11}+2y_{11}y_{12}\alpha_{12}+y_{12}^2\alpha_{22}$); it would be worth extracting a few posterior samples and checking which combinations are small. At minimum, the column ordering of the symbolic $U$ (GitHub) versus the PMNS mass-ordering convention needs re-examination — I suspect a column-permutation error between the symbolic output and Eq. (2.15)/(2.20).

The same issue afflicts Eq. (2.15) (lines 551–558): the text correctly notes the heaviest state is μ-aligned, but in the displayed matrix the μ row is dominantly in *column 1*, which under the standard ordering ($m_3$ heaviest = column 3) says the *lightest* is μ-aligned. Columns 1 and 3 appear interchanged.

### T2. Inverted-hierarchy section: $\theta_{23}$ cannot be fitted with $\epsilon\sim0.03$ and $\mathcal{O}(1)$ coefficients; and the scan prior contradicts the analytic estimate (§2.4, lines 964–1002)

From the paper's own Eq. for the angles (line 990), $\tan\theta_{23}\approx \epsilon u_{23}'$. The data require $\tan\theta_{23}\approx 1.1$; with $\epsilon\sim 0.03$ (Eq. (ep), line 973) this forces $u_{23}'\sim 37$ — well outside $\mathcal{O}(1)$ and indeed beyond the paper's own naturalness criterion $r\le30$. Numerical check confirms: with the texture $a_{11}=a_{13}=0$ and random $\mathcal{O}(1)$ couplings, $\theta_{23}\lesssim 1^\circ$ at $\epsilon=0.03$. The statement at line 992 ("if the $u^{(k)}_{ij}$ parameters are of order 1 and $\epsilon\sim0.03$, then we can fit... the data") is not right as written. Similarly $\tan\theta_{12}=1+\mathcal{O}(\epsilon)$ predicts $\theta_{12}\approx45^\circ$; reaching $33.7^\circ$ needs $\epsilon\,u\sim0.33$, impossible at $\epsilon=0.03$ with $\mathcal{O}(1)$ $u$.

Relatedly: line 997 assigns $\epsilon\sim\mathcal{U}[0.1,0.5]$ "based on our analytical arguments in (ep)" — but (ep) gives $\epsilon\sim0.03$, *outside* the prior range. Presumably the fit instead selects $\epsilon\sim0.36$ (the commented-out posterior at line 1106) with $B/A$ suppressed to maintain $\Delta m^2_{21}/|\Delta m^2_{32}|=0.03$. The analytic narrative and the scan set-up need to be reconciled; as it stands they contradict one another.

### T3. Sign error in the $\epsilon^2$ correction to $m_{\nu_3}$, Eq. (2.13) (lines 528–532)

Second-order perturbation theory gives
$$ m_{\nu_3} = \frac{v^2\epsilon^{-4}}{2M_0}\left[y_{22}^2\alpha_{22} \,{\color{red}+}\, \epsilon^2\left(2\alpha_{12}y_{21}y_{22} + \frac{(y_{12}\alpha_{22}+y_{11}\alpha_{12})^2}{\alpha_{22}}\right)\right], $$
i.e. the correction enters with a **plus** sign (the level-repulsion term $(X^{(1)}_{12})^2/\lambda_0$ necessarily pushes the large eigenvalue away from zero). Verified numerically at $\epsilon=0.02$: the plus-sign formula matches the exact eigenvalue to relative accuracy $10^{-6}$; the minus-sign version fails at $\mathcal{O}(\epsilon^2)$. The $m_{\nu_2}$ expression is correct as printed (verified to $3\times10^{-4}$).

### T4. Wrong seesaw formula at line 733

The text has $m_\nu = -\frac{v^2}{M_0}Y^T\Sigma^{-1}A^{-1}Y\Sigma^{-1}$. It should read
$$ m_\nu = -\frac{v^2}{2M_0}\,Y\,\Sigma^{-1}A^{-1}\Sigma^{-1}\,Y^T $$
— the factor $1/2$ is missing and the transposes/ordering are garbled ($\Sigma^{-1}$ must sit on both sides of $A^{-1}$, inside the $Y\cdots Y^T$ sandwich). Since this line defines what $M_0$ divides in the scan, please check the *code* uses the correct expression (the GitHub repository should be checked against this).

### T5. EFT consistency of the Majorana texture: $M_0\gg\Lambda_{\rm FN}$ needs justification (§2.2, Eq. on line 376)

The Majorana entries are written as $M_0\,a_{ij}\,\epsilon^{|X_i+X_j|}$ with $M_0\sim10^{13}$–$10^{17}$ GeV, while $\epsilon=\langle\theta\rangle/\Lambda_{\rm FN}$ with $\langle\theta\rangle\sim$ TeV. Two consistency questions arise:

1. The operator $\theta\,\overline{\nu^c_{R_1}}\nu_{R_3}$ is gauge invariant ($+1-1+0=0$) and **renormalisable**. With an $\mathcal{O}(1)$ coupling it yields $M_{13}\sim\langle\theta\rangle\sim$ TeV, not $M_0\,\epsilon\sim10^{14}$ GeV. Either that coupling must be argued absent/tiny, or the texture normalisation is not what the FN EFT gives.
2. More generally, FN mediators at $\Lambda_{\rm FN}$ generate $M_{ij}\sim\Lambda_{\rm FN}\,\epsilon^{|X_i+X_j|}$ (from $\theta^n\overline{\nu^c}\nu/\Lambda_{\rm FN}^{n-1}$), i.e. TeV-scale suppressed entries — a radically different matrix from $M_0\Sigma A\Sigma$, and one that would wreck the seesaw ($v^2/\text{TeV}$ is far too large). The implicit assumption is that the Majorana-sector mediators sit at $\sim M_0$, but then the relevant expansion parameter is $\langle\theta\rangle/M_0\sim10^{-12}$, not $\epsilon$.

The commented-out $S_{ij}$ construction (lines 893–954) gestures at a resolution; some version of it (or another argument) should appear in the published text, since at present the central texture (2.8) is not derived from the Lagrangian (2.7) as claimed — Eq. (2.7) itself writes the suppression as $(\langle\theta\rangle/\Lambda_{\rm FN})^{|X_i+X_j|}$ multiplying $M_0$, which is precisely the structure requiring justification.

### T6. $M_0$ estimate at line 548–549

Using $m_{\nu_3}\approx \frac{v^2\epsilon^{-4}}{2M_0}\times\mathcal{O}(1)$ (the *unsuppressed* Eq. (2.13), as the text says) with $\epsilon=0.42$ and $m_{\nu_3}=\sqrt{\Delta m^2_{31}}=0.05$ eV gives $M_0\approx2\times10^{16}$ GeV, not $10^{15}$. With the NH ($y_{22}=0$) scaling $\epsilon^{-2}$ one gets $3.5\times10^{15}$ GeV ✓. So either the reference to Eq. (2.13) or the quoted order of magnitude should change. Also "$m_{\nu_3}$ and $m_{\nu_1}$" should presumably be "$m_{\nu_3}$ and $m_{\nu_2}$" ($m_{\nu_1}=0$ carries no information).

### T7. IO PMNS matrix, lines 977–985: leading-order columns 1 and 2 are parallel

Column 1 is $(-1,1,\cdot)^T$ and column 2 is $(1,-1,\cdot)^T$ at leading order — the same vector up to sign, violating orthogonality. For a pseudo-Dirac pair from $X^{(0)}\propto\begin{pmatrix}0&c\\c&0\end{pmatrix}$ the eigenvectors are $(1,1)/\sqrt2$ and $(1,-1)/\sqrt2$; one column's relative sign is wrong.

### T8. Percent-level vs $\epsilon^2$-level suppression: internal inconsistency

The Introduction (line 253) and Conclusions (line 1114) say the NH requires "percent-level" suppression of one coupling; §2.3 (line 582 and the spread discussion at line 590) says $y_{22}=\mathcal{O}(\epsilon^2)\approx0.18$ — a factor-of-6 spread, i.e. tens-of-percent level. The clockwork example delivers 0.06. Decide which is actually required (this matters for the naturalness message) and make abstract, introduction, §2.3, appendix and conclusions agree. Note also line 1115–1117: for the IO it is two *Majorana* couplings ($a_{11},a_{13}$) that are suppressed, not "Yukawa couplings" (same slip in the abstract, "One or two Yukawa couplings").

### T9. Appendix spurion charges are inconsistent (lines 1385–1414)

With $\nu_i$ ($F_i$) carrying charge $-1$ under $U(1)_{A,B,C}$ respectively, the bilinear $\overline{\nu^c_i}\nu_j$ has charge $-(\delta_{iA}+\delta_{jA},\ldots)$; restoring $a_{22}$ needs a spurion of charge $(0,+2,0)$, $a_{12}$ needs $(+1,+1,0)$, etc. The listed charges $(-1,1,0),(0,-1,1),(0,-2,0),(0,0,-2)$ cannot compensate any Majorana bilinear (all bilinears have same-sign charges). Presumably a sign convention slipped; and at line 1412 "no spurion with charge $(-1,0,1)$ or $(0,0,2)$" — the second should be $(\pm2,0,0)$ (the $a_{11}$ spurion), since $(0,0,\mp2)$ *is* in the list (it generates $a_{33}$).

### T10. Appendix Lagrangians (A.5), (A.6): dimensions and chain structure

In Eq. (A.5) (line 1400) the terms $a_0\overline{\psi_{L0}}F_2^c$ and $a_1\overline{\psi_{R1}}F_2$ are mass-dimension-3 bilinears with apparently dimensionless couplings $a_0,a_1$ — a mass scale (or spurion VEV) is missing. The mixing term $\mu q\,\overline{\psi^c_{R1}}\psi_{R0}$ is Majorana-type (R–R), whereas (A.6) uses Dirac-type $\mu q\,\overline{\chi_{L0}}\chi_{R1}$ — is the asymmetry intended? Please check both against the underlying construction.

### T11. Smaller physics points

* **Line 213 ("dimension-4 Lagrangian density terms")**: the Majorana mass term is dimension 3. Say "renormalisable" instead.
* **Lines 227–231**: the right-hand column vector should be $(\nu_L^c,\ \nu_R)^T$ — barred spinors cannot form a column contracted from the right, and the $+\,$H.c. is missing.
* **Line 152, Eq. (1.3)**: exponent written $X_i-X_j$; later (line 376) $|-X_i+X_j|$. Use $|X_i-X_j|$ in both (and note that for negative net charge it is $\theta^\dagger$ insertions).
* **Line 468 claim** $P(r>30)=0.35$ for 9 unit normals: verified by Monte Carlo (0.353) ✓. Line 590 claim $P\approx0.9$ for spread $<6$... actually $P(r>6)=0.90$, so the probability of spread *at least* 6 is 0.9 — check the sentence's logic: as written ("easy to obtain... with a probability of .9") it is the probability of $r\ge6$, fine, but make the direction of the inequality explicit.
* **Naturalness baseline**: under the priors alone ($\mathcal{U}[-3,3]$), Monte Carlo gives $P(r<30)=0.79$ (8 couplings, NH) and $0.76$ (9, IO). The posterior values $68\pm2\%$ and $69\pm5\%$ are therefore *below* the prior baseline — the fit mildly disfavours natural configurations. Quoting the prior baseline alongside Eqs. (2.24), (2.32) would make the result interpretable.
* **Line 1090–1091 (IO table)**: for IO, NuFIT's $\Delta m^2_{3\ell}=\Delta m^2_{32}=-2.484\times10^{-3}\,$eV$^2$; the table lists "$\Delta m^2_{31}$" $=+2.48\times10^{-3}$. Label and sign need fixing (and units "/eV$^2$" are missing in this table, unlike Table 3).
* **Line 75**: "up to $4\sigma$" for $B_s\to\phi\mu\mu$ — LHCb quote $3.6\sigma$ in the $1.1<q^2<6\,$GeV$^2$ bin; consider "almost $4\sigma$" or the precise figure.
* **Line 1119**: "$U(2)_q\times U(2)_u\times U(2)_R$" — presumably $U(2)_d$ rather than $U(2)_R$?
* **Line 704**: "16-dimensional parameter space" — correct for NH (8 $y$ + 6 $a$ + $\epsilon$ + $M_0$) ✓, but the IO scan is 15-dimensional (9 $y$ + 4 $a$ + 2); if the text keeps a dimension count, keep it case-accurate.
* **Likelihood never defined**: nowhere is the likelihood function stated (Gaussian in the five NuFIT observables with the Table-2 $\sigma$'s, presumably, and how the ordering constraint is imposed). This must be specified for reproducibility, alongside the number of free parameters vs observables.

---

## 2. References

### Mis-attribution (should fix)

* **Line 1128**: "it had already been pointed out in Ref. [Bonilla:2017lsq] that $U(1)_{B_3-L_2}$ gauge symmetry..." — Bonilla, Modak, Srivastava & Valle propose $U(1)_{B_3-3L_\mu}$ (as their title says). Either change the symmetry name in the sentence, or cite the actual $B_3-L_2$ papers: Alonso, Cox, Han & Yanagida, [arXiv:1704.08158](https://arxiv.org/abs/1704.08158) (Phys. Rev. D 96 (2017) 071701) and Allanach & Davighi, [arXiv:2009.02197](https://arxiv.org/abs/2009.02197) (Eur. Phys. J. C 81 (2021) 56), both of which explain $b\to s\ell\ell$ with $B_3-L_2$-type symmetries and contain type-I seesaw remarks.

### Obvious candidates missing

* **LHCb 2020 angular analysis** of $B^0\to K^{*0}\mu^+\mu^-$, [arXiv:2003.04831](https://arxiv.org/abs/2003.04831), PRL 125 (2020) 011802 — the flagship $P_5'$ measurement; currently only the 2013 LHCb, ATLAS and CMS analyses are cited (line 79). Consider also $B^+\to K^{*+}\mu^+\mu^-$, arXiv:2012.13241.
* **Seesaw**: Schechter & Valle, Phys. Rev. D 22 (1980) 2227, usually cited alongside the four you have.
* **Clockwork**: Choi & Im, [arXiv:1511.00132](https://arxiv.org/abs/1511.00132) (JHEP 01 (2016) 149) is co-credited with Kaplan–Rattazzi as the origin of the mechanism; add at line 1145.
* **Neutrino anarchy / FN for neutrinos**: Hall, Murayama & Weiner, hep-ph/9911341 (PRL 84 (2000) 2572); optionally de Gouvêa & Murayama, arXiv:1204.1249. Natural companions to the anarchy discussion at line 1126.
* **Cosmology & $0\nu\beta\beta$ (recommended addition, see D4)**: DESI BAO bounds on $\Sigma m_\nu$ ([arXiv:2404.03002]; DR2 2503.14738; caveats in [arXiv:2406.14554](https://arxiv.org/abs/2406.14554)), Planck 2018 (arXiv:1807.06209), KamLAND-Zen (arXiv:2406.11438). The IO prediction ($\Sigma m_\nu\simeq0.10$ eV, $m_{\beta\beta}\sim20$–$50$ meV) is squarely testable and currently under pressure from some DESI combinations — the paper should say so.
* **Frampton, Glashow & Marfatia** (hep-ph/0201008, texture zeros of $m_\nu$) is *in* `article.bib` but never cited; the IO texture-zero discussion (§2.4) is a natural place for it, together with the two-zero-texture literature it spawned.

### Bibliography hygiene

* Duplicate entry: `Capozzi:2025wyn` and `PhysRevD.111.093006` are the same paper (only the former is cited — delete the latter).
* Uncited entries to prune (or cite): `deSalas:2020pgw`, `Frampton:2002yf`, `cauchy`, `gelman_prior_2006`, `polson_half-cauchy_2012`, `hoffman_no-u-turn_2011`, `phan_composable_2019`, `speagle_dynesty_2020`, `noauthor_nufit_nodate`, `casella_statistical_2002`, `coles_introduction_2001` (the last two are cited only inside `comment` blocks).
* Zotero-exported entries (`minkowski_e_1977`, `gell-mann_complex_2013`, `giudice_clockwork_2017`, `alonso_clockwork_2018`, `arkani-hamed_constructing_2001`, ...) carry abstracts/keywords/urldate and inconsistent metadata: e.g. Minkowski's title has "one out of 109" (should be $10^9$), and Gell-Mann–Ramond–Slansky is dated 2013 (the arXiv posting) rather than 1979. Strongly recommend replacing all of these with their Inspire-HEP entries (`Minkowski:1977sc`, `GellMann:1979vob`, `Giudice:2016yja`, `Alonso:2018gki`, `ArkaniHamed:2001ca`, `Esteban:2024eli`, ...) so JHEP-2.bst renders them uniformly with arXiv numbers.
* `github` entry: give it a proper title, both authors' correct initials ("Sarjapur, P.C." presumably, not "Prabhoda, S.C."), and ideally a tagged release/commit hash for reproducibility.

---

## 3. Copy-edit pass

Line-referenced; grouped. (I have not made any of these changes.)

**Front matter**

1. l.36: `\author[1]` — no affiliation "1" exists; should be `[a]` (as l.31). Also check whether jheppub wants the literal "and" inside `\author{}`.
2. Abstract, l.40/45: "Standard Model (SM)" defined twice.
3. Abstract & throughout: mixed $l^+l^-$ and $\ell^+\ell^-$ (macros use `\ell`). Standardise on $\ell$.
4. l.58 keywords and everywhere: `$B-$anomalies`, `$B-$meson` typeset a minus sign; use `$B$-anomalies`, `$B$-meson` (occurrences at l.58, 68, 1113, 1127...).

**Section 1**

5. l.140: sentence begins with bare "(\ref{chferms})" — prepend "Equation" or use `\cref`.
6. l.150: "the Higgs doublet field H" → `$H$`.
7. l.152: exponent $X_i-X_j$ → $|X_i-X_j|$ (see T11).
8. l.156: quote marks: ``spaghetti diagram" — closing double quote should be `''`. Check globally.
9. l.211: "right-handed neutrino type fields" → "right-handed-neutrino-type fields" or recast; "dimension-4" → "renormalisable" (see T11).
10. l.240: "...the mass matrix~\cite{...}." — full stop should follow the equation, not precede it; also order the seesaw citations chronologically and consistently with l.250 (currently two different subsets at the two mentions).
11. l.233: "6 by 6", l.240 "3 by 3" → "$6\times6$", "$3\times3$".

**Section 2**

12. l.312: "by virtue of them remaining invariant" → "by virtue of their invariance".
13. l.327: Eq. (2.7) lacks terminal punctuation; following "where" clause runs on.
14. l.342/403: `$\Lambda_{FN}$` (italic FN) vs `$\Lambda_{\text{FN}}$` elsewhere — standardise (`\text{FN}`).
15. Table 2: `$\Delta m_{3l}^2$` → `$\Delta m_{3\ell}^2$`; the table mixes booktabs rules (`\toprule`,`\midrule`) with a closing `\hline`; "Sept 2024" → "September 2024".
16. l.371: section-title capitalisation "Off-Diagonal entries" → "Off-diagonal entries" (match house style).
17. l.375–378: `\Tilde` → `\tilde`; bold-face `\bm{L}`, `\bm{\nu}` clash with earlier non-bold notation for the same fields; "H.c." styling differs from l.214 (`\textrm{H.c.}`).
18. l.378 etc.: `\(...\)` maths delimiters mixed with `$...$` (also l.382, 490–493). Standardise.
19. l.384/616: $Y_{\nu L}$ vs $Y_\nu$ earlier — one symbol, please; likewise $M$ (l.376) vs $M_R$ (l.857).
20. l.419: $\epsilon^1$ → $\epsilon$.
21. l.447: equation ends with ", " then a new sentence — punctuation.
22. l.483/749/783: $Y_{max}$, $Y_\mathrm{max}$, $Y_{\text{max}}$ all appear — standardise (`\text{max}`); same for min. Also $r<30$ vs $r\le30$ (l.487 vs l.770).
23. l.487: "We shall view $r$ less than a few tens then to be natural: here, we shall take $r\leq30$ to be definite." → "...to be natural; to be definite, we take $r\leq30$."
24. l.545: "$\frac{8\times10^{-5}}{2.5\times10^{-3}}$" — Table 2 says $7.49\times10^{-5}$; use consistent central values (or "$\approx8$").
25. l.559: "where $A$, $B$ and $C$ are rational functions..." — no $A,B,C$ appear in Eq. (2.15); delete or repair.
26. l.562–565: double colon construction ("...from Table 2: one expects a sizeable 23 mixing: the mixing...") — recast.
27. l.583, 586, 590: live `\bca{...}`/`\pcs{...}` notes in the main text (also l.1264, 1291–1292). All must be resolved and removed.
28. l.590: "a probability of $.9$" → "0.9"; also clarify inequality direction (see T11).
29. l.597: "in the level of effective theory" → "at the level of effective theory".
30. l.629 & 870: the symbols $X^{(0,1,2)}$ are silently redefined in §2.3 and again in §2.4 with different meanings from Eqs. (2.11–2.13); prime them or add "(redefined)".
31. l.651/889: $m$ used for the overall scale clashes with $m_i$ for masses (and is redefined between NH and IO with different $\epsilon$ powers); suggest $\bar m$ or $m_0^{\rm NH/IO}$.
32. l.655: stray extra ")" in $m_3=m(A_3+\epsilon^2B_3+\mathcal{O}(\epsilon^3)))$; the three eigenvalue equations lack punctuation.
33. l.670–671: Eq. uses $A_1$ (undefined — only $A_3$ exists) and $(B_3-B_2)^2$; from the eigenvalues it should be $\epsilon^4(B_2^2-B_1^2)/A_3^2$. Repair (this is also a technical slip).
34. l.684: sentence begins lowercase ("the entries $u^{(n)}_{ij}$ are long algebraic functions..."); "long" → "lengthy".
35. l.688: notation $u_{12}'$ introduced without definition alongside $u^{(1)}_{12}$; unify primes vs superscripts here and at l.988–991.
36. l.703–704: "methods such as Hamiltonian Monte Carlo (HMC), were considered" — spurious comma.
37. l.712: "using nested sampling using the Monte Carlo algorithm MLFriends" — "using... using"; also `\texttt{UltraNest}(v4.5.0) \footnote{...}` spacing before the footnote.
38. l.717: define PDF (and CDF, first used at l.768) at first use; "posterior PDF" l.717 vs "probability density functions" l.464.
39. l.728–730: "we get that a good estimate of $\epsilon<0.5$, naturally guiding us..." — grammar; also justify the lower end (0.1) of the prior.
40. l.737: "we fix $|y_{ij}|<3$ with the priors, $y_{ij}\sim\mathcal{U}[-3,3]$ and $a_{ij}\sim\mathcal{U}[-3,3]$ ... over the range $[-3,3]$" — redundant tail; the CP-phase footnote duplicates l.409.
41. l.749–750: stray spaces in `$y_{22}    $`.
42. l.767 vs 995: "$N_{\mathrm{live}}=50{,}000$" vs "$N_{\mathrm{live}}=50000$" — one format.
43. l.783 & 1062: identical figure captions; add "(normal ordering)"/"(inverted ordering)", full stops, and sentence-case ("Cumulative distribution function...").
44. l.786: "posterior means ... are presented in Fig. 4 and Fig. 5" — the figures show full posteriors, not means; recast. `corner(v2.2.3)` spacing.
45. l.787: "The posterior shown in $(\epsilon,\log_{10}M_0)$ plane" → "in the ... plane".
46. Table 3 caption: "1σ theoretical uncertainties" — they are posterior standard deviations; rename. Consider reporting $\epsilon$ and $\log_{10}M_0$ posteriors (currently only in a comment, l.1106–1107).
47. l.841 vs 855: "Setting ... we get an inverted mass hierarchy" and "We now show that with these texture zeroes, we obtain an inverted hierarchy" — repetitive.
48. l.889: comma splice "as in §2.4, we obtain".
49. l.969: "i.e. an inverted hierarchy (i.e. $m_3\ll m_1<m_2$)" — double "i.e.".
50. l.996–1001: "once again" three times in six lines.
51. l.1029: "all converge to the NuFIT-6.0 best fit values" → "agree with"; `NuFIT-6.0` vs `{\tt NuFIT 6.0}` formatting.
52. Tables 3/4: `|c|c|c|` ruled style vs booktabs in Table 2 — one style throughout.
53. l.1113: "put to three uses: to explain..., to explain... and to suppress" — fine, but "measurements involving the $b\to s\ell\ell$ transition" vs macro `\bsll` — use the macro.
54. l.1133–1134: "fits to measurements involving $B$ mesons are fit significantly better" — "fits...are fit"; recast.
55. l.1146: "...clockwork mechanism \cite{...}, In our slightly modified setup..." — comma should be a full stop; "In" capitalised mid-sentence at l.1458 too ("We note that In Ref.").
56. l.1147–1150: "we assume heavy ``gears'' or an $N$-node chain" — the chain has $N+1$ nodes (cf. l.1249); recast the sentence (it currently reads as though kinetic terms are neglected *because* the gears are heavy — say "whose heavy mass eigenstates we integrate out").
57. l.1266: "path/chain length" — choose one; define $n(BB')$ notation properly.
58. l.1285: `\mathcal{L}_{eff}` → `\mathcal{L}_{\text{eff}}`.
59. l.1294/1436: "texture zeros" vs l.586 "texture zeroes" — standardise.
60. l.1425: "here, the overall factor..." begins lowercase after a displayed equation ending in a full stop.
61. Figures: paths point to `../Python/Data/...` — for arXiv submission the PNGs must be brought into the LaTeX directory; also enlarge axis-label fonts on the corner plots and explain the dashed lines in the captions.
62. Residual `#neut.tex#` and `*~` emacs artefacts in the directory — harmless, but exclude from any arXiv tarball.

---

## 4. Deeper editorial pass

**D1. Restructure §2's numerics to avoid duplication.** The Bayesian methodology (sampler, priors, $N_{\rm live}$, $r$-statistic) is described twice, once per ordering (lines 700–749 and 994–1013), with near-identical wording. Pull it into a single "Numerical methodology" subsection (priors table, likelihood definition — currently absent, see T11 — sampler settings, convergence diagnostics), then two short results subsections. This would shorten the paper and eliminate the "once again" litany.

**D2. Make the analytic sections honest about what they show.** Given T1/T2, I suggest reframing §§2.3–2.4: the analytic expansions demonstrate that the *mass hierarchies* come out correctly ($m_2/m_3\sim\epsilon^2$; pseudo-Dirac pair for IO), while the *angles* require particular $\mathcal{O}(1)$ combinations to be mildly enhanced/suppressed — which the global fit then quantifies via the spread $r$. That narrative is defensible and matches your numerics; the current "O(1) coefficients fit everything" claim is not. It would also be illuminating to show *which* combinations the posterior suppresses (e.g. 1D posteriors of $y_{11}$, $X^{(0)}_{11}$, $B/A$).

**D3. Interpret the naturalness numbers against the prior baseline.** $P(r<30)=68\%$/$69\%$ reads as a success, but the prior alone gives 79%/76% (Monte Carlo, §1 T11). The honest statement is that fitting the data costs only ~10 percentage points of naturalness — arguably a *stronger* selling point, but it must be phrased relative to the baseline. Consider also quoting the analogous figure from Ref. [Allanach:2026rkb]'s normal-distribution measure for consistency with §2.2's set-up (the priors are uniform, the naturalness discussion is framed with unit normals — the two notions of $\mathcal{O}(1)$ should be reconciled or the difference acknowledged).

**D4. Add a "predictions and tests" paragraph (probably in the Conclusions).** The model is falsifiable and the paper currently doesn't say how. With $m_{\rm lightest}\approx0$: NO gives $\Sigma m_\nu\approx0.059$ eV and $m_{\beta\beta}\sim1$–4 meV (hard); IO gives $\Sigma m_\nu\approx0.10$ eV and $m_{\beta\beta}\approx18$–48 meV — within reach of LEGEND-1000/nEXO, and in tension with the more aggressive DESI+CMB combinations. Since one of the paper's headline results is that IO is the *more natural* ordering, the observation that cosmology is currently squeezing IO deserves a sentence and citations (see §2). The heavy-neutrino spectrum ($M_3\sim M_0$, $M_1\sim\epsilon^2M_0$, $M_2\sim\epsilon^4M_0$) and a remark on leptogenesis prospects (real couplings assumed here ⇒ no CP violation — worth stating explicitly as a limitation) would round this out.

**D5. Address the $M_0$ vs $\Lambda_{\rm FN}$ scale question head-on (T5).** Even a paragraph acknowledging that the Majorana texture presumes FN mediators in the $\nu_R$ sector near $M_0$ (with its own justification, or the resurrected $S_{ij}$ construction) would pre-empt the most obvious referee objection to the set-up.

**D6. Symbol table / notational tidy.** By the appendix, the reader has met $m$, $\mu$, $m_0$, $M$, $M_R$, $M_0$, $m_\nu$, $m_{\nu L}$, $m_{\nu D}$, and three different meanings of $X^{(i)}$. A short notation paragraph (or simply pruning, per copy-edit items 19, 30, 31) would help.

**D7. GitHub citation and reproducibility.** Since the "detailed forms of the eigenvalues" and all numerics live in the repository, pin a release, state the UltraNest random seeds/convergence criteria, and list package versions in one place (you already do this for UltraNest and corner — add numpy/scipy/python).

**D8. Consider a frequentist cross-check.** (Echoing the in-source comment at line 1046.) Since the system is under-constrained, the posterior "agreement" with data in Tables 3–4 is guaranteed and carries little information (as the text candidly admits at line 836). A profile-likelihood check that the best-fit $\chi^2\approx0$ in each ordering, plus the $r$-distribution, is really all that matters; consider demoting Tables 3–4 to confirmation status and promoting the $r$/naturalness comparison NH vs IO to the headline result, including a Bayes-factor comparison of the two orderings (nested sampling gives you the evidences essentially for free — quote $\ln Z_{\rm NO}-\ln Z_{\rm IO}$!).

---

*Verification notes: items T1–T4 and the two Monte Carlo statements (P(r>30)=0.35 for 9 unit normals; prior baselines 0.79/0.76) were checked numerically with exact diagonalisation / 2×10⁶-sample Monte Carlo. The NuFIT 6.0 numbers in Table 2 were checked against the published values (IC24 with SK-atm) and are correct.*
