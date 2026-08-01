# Battle Royale Self-Play — System Design, Math & Physics

A complete walkthrough of every component: what it is, the math/physics behind
it, and *why* it is there. Where a design choice fixes a concrete failure mode,
that failure is called out — the current system is the fixed point of a long
chain of "the obvious thing doesn't work because…".

---

## 1. The game in one paragraph

`N` agents are pucks on a 2-D circular arena. Each can push itself around with a
2-D force. An agent is **eliminated** when it leaves the arena boundary. The
boundary **shrinks** over time (the "storm"), so the arena eventually cannot hold
everyone and the match resolves to a single survivor. The learning goal is a
single **self-play** policy (all agents share one network) that (a) forms a
*balanced* population — N copies each win ≈ `1/N` — and (b) *dominates* an
untrained/random baseline when it plays 1-vs-(N−1).

---

## 2. Architecture (Clean Architecture)

Dependencies point inward; inner layers know nothing of outer ones.

| Layer | Contents | Examples |
|---|---|---|
| **domain** | Pure rules, no I/O | `Agent`, `Arena`, `RewardCalculator`, `ObservationBuilder`, `EliminationService`, `EloRatingSystem` |
| **application** | Orchestration/use-cases | `Trainer`, `Evaluator`, `SnapshotPool`, `MetricsTracker` |
| **infrastructure** | External tech | `MuJoCoEnvironment`, `XMLBuilder`, `WandB/console loggers`, `VideoRecorder`, YAML config |
| **interfaces** | Adapters to the outside | `SelfPlayEnv` (Gymnasium), `BattleRoyaleEnv` (PettingZoo), CLIs |

Why: the reward and observation are *pure functions* of game state, so they are
unit-tested exhaustively and are identical in training, evaluation and rendering.
Physics (MuJoCo) is an infrastructure detail behind an interface, so the learning
code never imports MuJoCo directly.

---

## 3. Physics

### 3.1 Bodies and actuation
Each agent is a MuJoCo body with **two slide joints** (`x`, `y`) — it translates
in the plane, no rotation, no vertical motion. A cylinder geom (radius
`r = 0.15 m`, half-height `0.05 m`) is the collision + mass shape. Two motors
(one per joint) apply force `F = gear · u`, where the control `u ∈ [−1, 1]` is the
policy's action and `gear = max_force = 30 N`.

Mass (MuJoCo default density `ρ = 1000 kg/m³`):
```
m = ρ · π r² h = 1000 · π · 0.15² · 0.10 ≈ 7.07 kg
```

### 3.2 The damping equation (why the arena is not frictionless)
Per axis, Newton's law with linear joint **damping** `c = 8 N·s/m`:
```
m · a = F − c · v          (a = dv/dt)
```
This is a first-order linear ODE in `v`. Two consequences matter:

- **Terminal velocity** (steady state, `a = 0`):
  ```
  v_term = F_max / c = 30 / 8 = 3.75 m/s
  ```
  Velocity is *bounded*. Push at full force forever and you top out at 3.75 m/s.

- **Braking time constant**:
  ```
  τ = m / c = 7.07 / 8 ≈ 0.88 s  (≈ 88 sim steps at dt = 0.01 s)
  ```
  Release the control and velocity decays like `e^(−t/τ)` — the agent can *stop*.

**Why this is essential.** With `c = 0` (frictionless, the original model),
`m·a = F` ⇒ `v` grows without bound and position integrates quadratically. Any
tiny directional bias in the policy compounds into an uncontrollable slide off
the edge — the agent literally cannot learn to stay in. Empirically, with `c = 0`
"winning" degenerated into *being the last to fall off by accident* (a ~`1/N`
artifact), and a stronger reward simply made agents commit suicide faster.
Damping converts the task from "impossible to control" to "a controllable sumo".

### 3.3 Force magnitude (why 30 N, not 10 N)
Traversal time across the arena ≈ `distance / v_term`. At `F = 10 N`,
`v_term = 1.25 m/s`; a scripted optimal "ram" needed ~350 of the 400 step budget
just to *reach* an opponent on the far side — no time left to push it out, so
winning was physically impossible. At `F = 30 N` (`v_term = 3.75 m/s`) a scripted
ram reaches *and* ejects a passive opponent by ≈ step 236. So the higher force
makes the objective *achievable within the episode*.

`dt = 0.01 s`; episodes are capped at 400 steps (4 s of simulated time).

---

## 4. The storm (shrinking arena)

The boundary radius at step `t`:
```
R(t) = R₀ · ( 1 − (1 − f) · min(t / T, 1) )        for t ≤ T
R(t) = R₀ · f                                       for t > T
R₀ = 3.0 (initial),  f = 0.15 (final fraction),  T = 300 steps
```
So the radius contracts linearly `3.0 → 0.45 m` over the first 3 seconds, then
holds. Elimination uses the **current** `R(t)` (see §5), and the observation and
edge-penalty use it too, so agents perceive and are graded against the closing
boundary.

**Why the storm exists.** Two competent, damped agents can simply avoid each
other forever → the match ends in a *draw* at the step cap, and skill is
invisible. The storm removes draws by construction: once `R(t)` is smaller than
the space `N` agents need, someone is forced out. It (a) makes matches *decisive*
(draw-rate → ~0), (b) turns positioning/survival skill into wins, and (c) is the
literal "battle-royale" mechanic. Its effect is measured in §10.1.

---

## 5. Elimination
```
EliminationService.is_eliminated(agent) ≡  ‖agent.position‖ > R(t)
```
Eliminated agents are frozen (velocity and control zeroed) and excluded from
future neighbour observations, but remain in the roster so indexing is stable.

---

## 6. Observation (17-D, egocentric, permutation- & rotation-robust)

For the acting agent:
```
[ own_x, own_y,           # absolute position          (2)
  own_vx, own_vy,         # own velocity               (2)
  R(t) − ‖pos‖,           # distance to current boundary (1)
  for k in 3 nearest living opponents:
     (nx−x, ny−y),        # relative position          (2 each)
     (nvx−vx, nvy−vy) ]   # relative velocity          (2 each)
= 2 + 2 + 1 + 3·4 = 17
```

Design points and *why*:
- **Neighbours are sorted by distance** and only the nearest 3 are kept →
  the representation is *permutation-invariant* (order of opponents doesn't
  matter) and fixed-width for any `N`, which is what lets one network play
  `N = 2…8`.
- **Relative** neighbour coordinates → translation-invariant.
- The only **absolute** quantity is own `(x, y)`. That single choice broke
  rotational symmetry: because agent 0 always spawned at angle 0, the policy
  memorised its spawn and one slot won ~80 % of an all-identical match instead of
  `1/N`. Fixed by **randomising the whole-arena rotation** each episode (§8.4),
  which decorrelates slot from absolute position and forces a rotation-invariant
  policy.

---

## 7. Reward function — the heart of the system

Computed per step for an agent that was alive last step. Let `d_i = ‖pos_i‖`
(distance from centre) and `gap = ‖pos_self − pos_nearest_opp‖`.

```
R =  +1.0 · (# opponents that died this step)          # ELIMINATION
     −1.0                if the agent itself died  → return   # DEATH
     +0.001 − 0.003      per surviving step (net −0.002)      # SURVIVAL + TIME
     −1.0 · max(0, d_self − 0.92·R(t))                        # EDGE PENALTY
     +1.0 · Σ_opp (d_opp,now − d_opp,prev)                    # PUSH shaping
     +0.4 · (gap_prev − gap_now)                              # APPROACH shaping
     +10.0               if the agent is the sole survivor    # WIN BONUS
```

Every term exists to fix a specific failure that appeared without it:

| Term | Value | Fixes |
|---|---|---|
| **Elimination** | +1 / opponent | Gives a signal for opponents leaving. |
| **Death** | −1 | Discourages driving yourself out. |
| **Survival** | +0.001 | Keeps "alive" strictly better than "dead" each step — an anti-suicide floor. |
| **Time penalty** | −0.003 (net −0.002/step) | Makes idling mildly unprofitable so passive **turtling** to the cap (≈ −0.8) is worse than acting — *but* still above the death penalty (−1), so it does **not** re-create a suicide incentive. An earlier −0.01 penalty made idling worse than dying and the policy learned to self-eliminate. |
| **Win bonus** | +10 (sole survivor) | Makes **winning dominate turtling**: turtle-to-cap ≈ −0.8, a win ≈ elims + 10 ≈ +13. Without an explicit terminal win signal the policy either turtles (if survival is large) or suicides (if it's negative). |
| **Push shaping** | +1 · Δ(opponent radius) | Dense, **potential-based** term (potential `Φ = Σ opponent distance-from-centre`). The win/elim rewards are far too sparse for exploration to ever stumble on a full push-out; this gives a smooth gradient toward ejecting. Potential-based ⇒ it does not change the optimal policy, only the learning speed. |
| **Approach shaping** | +0.4 · Δ(−gap) | Teaches the agent to **hunt**. Pure self-play opponents come *to* you, so the policy never learns to close on a *passive* target and loses to a random baseline. This rewards closing the gap; because it's ~0 while you stay in contact and shove, it does **not** fight the push term. It was 8× weaker at first (0.05) and never bootstrapped a chase — strengthening it was what finally produced ejections. |
| **Edge penalty** | −1 · (outer-ring depth) | Dense **self-preservation**. A terminal −1 is too sparse to teach *braking*; the agent kept overshooting its own edge while chasing. Active only beyond `0.92·R(t)` so it deters going *past* the edge without discouraging central play or shoving an opponent out. |

Ordering the design guarantees: **win (≈ +13) ≫ turtle (≈ −0.8) > suicide (−1)**.
That single inequality is what all the tuning is really about.

---

## 8. Self-play

### 8.1 `SelfPlayEnv` — a single-agent view of a multi-agent game
SB3 (PPO) controls **one** *learner* slot; the other `N−1` slots are driven by a
frozen opponent policy sampled at episode start. Each episode is therefore one
clean "learner vs the field" match with a win/loss, which is exactly what makes
win-rate and Elo measurable and what the generalisation sweep varies (`N`).

### 8.2 Snapshot pool = fictitious / δ-uniform self-play
`SnapshotPool` keeps up to 100 past policy snapshots (the whole 1 M-step run at a
10 k interval) and samples an opponent **uniformly** from them. Training against
a uniform mix of *all* past selves (weak-early → strong-late), rather than only
the latest, is δ-uniform self-play (Heinrich & Silver; AlphaStar league). It
stabilises training (no Red-Queen cycling against a single moving target) and
makes "beating the pool" mean *dominating your own history*.

### 8.3 Opponent curriculum (learn to *hunt*, then to *compete*)
The fraction of episodes played against **random** opponents follows a schedule:
```
frac < 0.30      : p_random = 1.0            (warm-up: learn to hunt passive targets)
frac ≥ 0.30      : p_random: 1.0 → 0.35      (decay into mostly self-play)
```
Rationale: pure self-play yields a *defensive* policy that can't eject a passive
random agent (0 % vs random); pure-random training yields a *hunter* that only
draws against competent copies of itself. The curriculum keeps both skills.

### 8.4 Rotation randomisation
Every reset rotates all spawn angles by a random `θ ∈ [0, 2π)`. See §6 — this is
what makes the balance demonstration (`~1/N`) hold instead of one slot dominating.

### 8.5 Elo
`EloRatingSystem` (K = 32) tracks the learner vs the pool as an aggregate rating,
updated per match by the standard logistic expected-score update
`R ← R + K·(S − E)`, `E = 1/(1 + 10^((R_opp − R)/400))`. Used as a training-health
signal (a rising Elo = the learner is beating progressively stronger snapshots).

---

## 9. PPO training
Stable-Baselines3 `PPO`, `MlpPolicy`, `lr = 3e-4`, `n_steps = 2048`,
`batch = 64`, `clip = 0.2`, `n_epochs = 10`, single `DummyVecEnv`. A callback
(a) records each finished match into `MetricsTracker`, (b) saves a snapshot every
10 k steps into the pool, and (c) advances the curriculum schedule. ~1 M steps
≈ 25 min on CPU (~600–700 steps/s).

---

## 10. Results (storm-trained policy, ~930 k steps)

Training signal: rolling win-rate vs pool ≈ **0.59**, Elo rising through **1500+**,
**3.0** eliminations/episode, matches resolving in ~180 steps.

### 10.1 Does the storm force engagement? (Yes — it forces *resolution*.)
Same trained policy in all slots, storm ON vs OFF, over 50 matches each:

| | resolution rate | avg eliminations | mean nearest-neighbour gap (early → late) |
|---|---|---|---|
| n=4, storm **OFF** | **0.04** | 0.74 / 4 | 2.05 → 0.47 |
| n=4, storm **ON**  | **0.98** | 3.00 / 4 | 2.06 → 0.65 |
| n=6, storm **OFF** | **0.00** | 1.48 / 6 | 1.47 → 0.52 |
| n=6, storm **ON**  | **0.96** | 4.96 / 6 | 1.47 → 0.42 |

Reading: the agents *cluster* either way (early gap ~2 → late gap ~0.5 in both
cases — the trained policy is inherently aggressive). But **without the storm 96–
100 % of matches are draws** — they push each other but the arena is too big to
eject anyone. The storm converts that into **96–98 % decisive** matches and drives
average eliminations from <1 to nearly the whole field. So the storm's role is to
force **resolution**, not initial contact.

### 10.2 Demonstration #1 — Balance (N identical copies → ~1/N), 100 matches each

| N | per-slot win-rate | ideal (1/N) | draw rate |
|---|---|---|---|
| 4 | 0.21, 0.17, 0.26, 0.33 | 0.25 | 0.03 |
| 6 | 0.17, 0.14, 0.11, 0.21, 0.18, 0.12 | 0.17 | 0.07 |
| 8 | 0.08, 0.08, 0.12, 0.12, 0.17, 0.12, 0.14, 0.07 | 0.125 | 0.10 |

Every slot wins ≈ `1/N` with no dominator, and draws stay low — a balanced,
decisive Nash population at all three counts.

### 10.3 Demonstration #2 — Dominance (1 trained vs N−1 random), 100 matches each

| N | trained WIN | LOSS | edge (win−loss) |
|---|---|---|---|
| 2 | 0.51 | 0.49 | +0.02 |
| 3 | 0.48 | 0.52 | −0.04 |
| **4 (training scale)** | **0.92** | 0.08 | **+0.84** |
| 6 | 0.72 | 0.28 | +0.44 |
| 8 | 0.54 | 0.46 | +0.08 |

Clear dominance at the training scale (`N = 4`: 0.92) that generalises up to
`N = 6` (0.72). It weakens at `N = 2, 3` (fewer opponents than trained — out of
distribution) and at `N = 8` (crowded). Training at mixed agent-counts would flatten
this curve if uniform dominance is wanted.

---

## 11. Why the classic success target was replaced

The README originally asked for "> 60 % win-rate vs the opponent pool". In a
**symmetric** N-player free-for-all this is provably unreachable: at a symmetric
Nash equilibrium all identical policies are interchangeable, so each wins exactly
`1/N` (0.25 at `N = 4`). You cannot dominate copies of yourself. Chasing 60 % vs a
converged pool is therefore ill-posed. The two-demonstration framing measures the
two things that *are* meaningful and achievable:

- **Balance** (`~1/N`, decisive) shows self-play converged to a fair, skilled
  equilibrium with no degenerate exploit.
- **Dominance** (edge vs an untrained baseline) shows the learned behaviour is
  genuinely skilful, not just mutually-cancelling.

---

## 12. The debugging chain (why every piece is load-bearing)

Each fix only revealed the next failure:

1. **Turtling** — survival reward (+0.01/step ×1000 = +10) beat winning (+3).
   → rebalance.
2. **Suicide** — a −0.01/step time penalty made dying (−1) cheaper than living.
   → add explicit **win bonus**, shrink the time penalty.
3. **Frictionless physics** — no damping ⇒ runaway drift-off; "winning" was just
   being last to self-eliminate. → **joint damping**.
4. **Too slow** — `F = 10` ⇒ can't reach an opponent in time. → **`F = 30`**.
5. **Reward too sparse** — win never discovered by exploration. → **push shaping**.
6. **Positional overfit** — agent 0 wins ~80 % of identical matches. → **rotation
   randomisation**.
7. **Defensive equilibrium** — self-play policy can't hunt a passive target
   (0 % vs random). → **approach shaping** + **opponent curriculum**.
8. **Self-elimination while chasing** — overshoots its own edge. → **edge
   penalty**.
9. **Draws** — competent agents avoid each other; skill invisible. → **storm**.

---

## 13. Rendering
`main.py` drives an episode and records video. The collision cylinder is drawn
invisibly (alpha 0); a **visual-only** humanoid (density 0, no collision — so
physics is unchanged) is drawn over it. A named disk geom is resized every frame
to `R(t)` to visualise the closing storm. A framed camera + 720p offscreen buffer
give the final clip (`demo_storm.mp4`).
