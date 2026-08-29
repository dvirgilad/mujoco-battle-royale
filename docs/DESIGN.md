# System Design — logic, physics & the reasoning behind each choice

A component-by-component walkthrough of the final system: **what** each part is,
the **math/physics** behind it, and **why** it is there. Where a choice fixes a
concrete failure, that failure is named — the system is the fixed point of a long
chain of *"the obvious thing doesn't work because…"*.

> This doc describes the **final** system, configured by `config/sumo_light.yaml`.
> For the full research narrative — every experiment, dead end, and the payoff
> algebra behind the fixes — see [`METHODOLOGY.md`](METHODOLOGY.md). For run
> commands see [`RUNBOOK.md`](RUNBOOK.md).

---

## 1. The game in one paragraph

`N` agents (4–8) are pucks on a 2-D circular arena. Each pushes itself around with
a 2-D force. An agent is **eliminated** when it leaves the arena boundary; the
last one in wins. All agents share **one** self-play policy. Success is three
measurable behaviours, all in the hard **melee** regime (n ≥ 4):

1. **Balance** — N identical copies each win ≈ `1/N`.
2. **Dominance** — 1 trained agent beats N−1 untrained/random agents.
3. **Active pushing** — agents win by *shoving opponents out*, not by outlasting a
   shrinking boundary.

The headline policy achieves all three at n = 4 and n = 6 (see §10).

---

## 2. Architecture (Clean Architecture)

Dependencies point inward; inner layers know nothing of outer ones.

| Layer | Contents | Examples |
|---|---|---|
| **domain** | Pure rules, no I/O | `Agent`, `Arena`, `RewardCalculator`, `ObservationBuilder`, `EliminationService`, `EloRatingSystem` |
| **application** | Orchestration / use-cases | `Trainer`, `Evaluator`, `SnapshotPool`, `MetricsTracker` |
| **infrastructure** | External tech | `MuJoCoEnvironment`, `XMLBuilder`, WandB/console loggers, `VideoRecorder`, YAML config |
| **interfaces** | Adapters to the outside | `SelfPlayEnv` (Gymnasium), `BattleRoyaleEnv` (PettingZoo), CLIs |

**Why.** The reward and observation are *pure functions* of game state, so they
are unit-tested exhaustively and are byte-identical across training, evaluation
and rendering. Physics (MuJoCo) is an infrastructure detail behind an interface,
so the learning code never imports MuJoCo directly — swapping the simulator or the
RL algorithm touches one layer only.

---

## 3. Physics

### 3.1 Bodies and actuation
Each agent is a MuJoCo body on **two slide joints** (`x`, `y`) — it translates in
the plane; no rotation, no vertical motion. A cylinder geom (radius `r = 0.15 m`,
half-height `0.05 m`) is the collision + mass shape, drawn invisibly; a
visual-only humanoid (zero mass, no collision) is drawn over it for rendering.
Two motors apply force `F = gear · u`, control `u ∈ [−1, 1]`, `gear = max_force = 14 N`.

Mass depends on the cylinder **density** `ρ` (exposed as `arena.agent_density`):
```
m = ρ · π r² (2·half_height) = ρ · π · 0.15² · 0.10
   ρ = 1000 (MuJoCo default) → m ≈ 7.07 kg     (original, heavy)
   ρ = 150  (sumo_light)     → m ≈ 1.06 kg     (final, light)
```

### 3.2 The damping equation — bounded speed *and* the braking-distance trap
Per axis, Newton's law with linear joint **damping** `c = 4 N·s/m`:
```
m · v̇ = F − c · v
```
Three quantities follow, and the **third is the load-bearing insight of the whole
project**:

- **Terminal velocity** `v∞ = F/c = 14/4 = 3.5 m/s` — independent of mass; push
  forever and you top out here. Damping is what makes the arena *not* frictionless:
  with `c = 0`, `v` grows unbounded and any bias slides you off the edge — the task
  is literally unlearnable.
- **Braking time-constant** `τ = m/c`. Cut thrust and `v(t) = v₀ e^{−t/τ}`.
- **Braking distance** from terminal velocity, `d_brake = v∞ · τ = F·m / c²`:
  ```
  heavy (m=7.07): d_brake = 14·7.07 / 16 ≈ 6.2 m   (≈5.5 m with reverse thrust)
  light (m=1.06): d_brake = 14·1.06 / 16 ≈ 0.93 m   (measured 0.96 m)
  ```

**Why this decides everything.** The arena radius is **~1.2 m**. With heavy
agents `d_brake ≫ R_arena`: an agent that reaches any real speed *physically cannot
stop inside the ring*. That single inequality forces **both** failure modes we
chased for a dozen experiments — self-ejection (overshoot the edge and die) and
caution (never commit to a push) — and **no reward change can defeat it**, because
it is a hard dynamical constraint. Lowering density to `ρ = 150` makes
`d_brake < R_arena`; `v∞` is unchanged and lighter opponents are displaced *more*
per impulse, so pushing gets easier at the same time. This is the change that
turned a −0.26 dominance ceiling into +0.98. Full derivation: METHODOLOGY §3.1.

### 3.3 Force magnitude and horizon
At `F = 14 N`, `v∞ = 3.5 m/s`; an agent can cross the ~2.4 m arena in well under a
second, leaving most of the episode to fight. `dt = 0.01 s`; episodes are capped at
`episode_max_steps = 600` (6 s of simulated time).

---

## 4. The arena boundary — storm optional, off by default

The boundary radius at step `t` is
```
R(t) = R₀ · (1 − (1 − f)·min(t/T, 1)),   f = min_radius_frac,  T = shrink_steps
```
Setting `f = 1` (or `T = 0`) makes `R(t) ≡ R₀` — a **fixed** arena. **The headline
policy runs with no storm** (`min_radius_frac: 1.0`), on purpose: with no closing
boundary to eliminate anyone "for free", **pushing an opponent out is the only way
to win**, so every resolution is a genuine push-out and the *active-pushing* target
is meaningful rather than a survival timer.

The storm is retained as a mechanism because it was instructive: it trivially
removes draws (once `R(t)` is too small for `N` agents, someone is forced out) and
was how earlier storm-trained policies reached balance/dominance — but it *replaces*
pushing rather than causing it, and rewards non-contact wins, the opposite of goal
(3). See METHODOLOGY §4, run #3.

---

## 5. Elimination
```
EliminationService.is_eliminated(agent) ≡  ‖agent.position‖ > R(t)
```
Eliminated agents are frozen (velocity and control zeroed) and dropped from future
neighbour observations, but stay in the roster so slot indexing is stable.

---

## 6. Observation (17-D, egocentric, permutation- & rotation-robust)

For the acting agent:
```
[ own_x, own_y,            # absolute position            (2)
  own_vx, own_vy,          # own velocity                 (2)
  R(t) − ‖pos‖,            # distance to current boundary (1)
  for k in 3 nearest living opponents:
     (nx−x, ny−y),         # relative position   (2 each)
     (nvx−vx, nvy−vy) ]    # relative velocity   (2 each)
= 2 + 2 + 1 + 3·4 = 17
```

Design points and *why*:
- **Nearest-3, sorted by distance** → fixed-width and *permutation-invariant*, which
  is exactly what lets one network play any `N = 2…8`.
- **Relative** neighbour coords → translation-invariant.
- The only **absolute** quantity is own `(x, y)`. That single choice broke rotational
  symmetry: because `agent_0` always spawned at angle 0, an early policy memorised its
  spawn and one slot won ~80 % of an all-identical match instead of `1/N`. Fixed by
  **randomising the whole-arena rotation** each episode (§8.4).
- *Residual bias:* on a regular n-gon the two adjacent neighbours are **exactly
  equidistant**, so the stable sort tie-breaks by agent index, giving each slot a
  fixed CW/CCW "first neighbour" and a small per-slot win bias. Attempts to remove it
  (spawn jitter, neighbour-shuffle) collapsed the aggressive behaviour — a documented
  fragility result, METHODOLOGY §3.7.

---

## 7. Reward function — the heart of the system

Computed per step for an agent that was alive last step. `d_i = ‖pos_i‖` (distance
from centre), `gap = ‖pos_self − pos_nearest_opp‖`.

```
R =  +1.0 · (# opponents that died this step)                 # ELIMINATION
     −1.0                if the agent itself died  → return    # DEATH
     +0.001 − 0.003      per surviving step (net −0.002)       # SURVIVAL + TIME
     −1.0 · max(0, d_self − 0.92·R(t))                         # EDGE PENALTY
     + a · 2.0 · Σ_opp (d_opp,now − d_opp,prev)                # PUSH shaping
     + a · 0.4 · (gap_prev − gap_now)                          # APPROACH shaping
     +10.0               if the agent is the sole survivor     # WIN BONUS
```
`a = aggression_scale` (annealed; §8.6). On a timed-out **draw** the learner is
additionally charged `−draw_penalty` (§8.5).

| Term | Value | Fixes |
|---|---|---|
| **Elimination** | +1 / opponent | a signal for opponents leaving |
| **Death** | −1 | discourages driving yourself out |
| **Survival** | +0.001 | "alive" strictly beats "dead" each step — an anti-suicide floor |
| **Time penalty** | −0.003 (net −0.002/step) | idling to the cap (≈ −1.2) is mildly unprofitable — but still above death (−1), so it does *not* re-create a suicide incentive |
| **Win bonus** | +10 (sole survivor) | makes **winning dominate turtling**: turtle ≈ −1.2, win ≈ elims + 10 ≈ +13 |
| **Push shaping** | +2.0 · Δ(opponent radius) | dense, **potential-based** (`Φ = Σ opp distance-from-centre`); the win/elim signal is far too sparse for exploration to stumble on a push-out. Potential-based ⇒ optimal policy unchanged (Ng et al. 1999) |
| **Approach shaping** | +0.4 · Δ(−gap) | teaches the agent to **hunt** a passive target; pure self-play opponents come *to* you, so without it the policy can't eject a *random* baseline (0 % vs random) |
| **Edge penalty** | −1 · (outer-ring depth) | dense **self-preservation**: a terminal −1 is too sparse to teach braking; active only beyond `0.92·R(t)` so it deters overshooting without discouraging shoves |

The whole design guarantees one inequality: **win (≈ +13) ≫ turtle (≈ −1.2) > suicide (−1)**.

---

## 8. Self-play

### 8.1 `SelfPlayEnv` — a single-agent view of a multi-agent game
PPO controls **one** *learner* slot; the other `N−1` slots run a frozen opponent
policy sampled at episode start. Each episode is therefore one clean
"learner-vs-field" match with a win/loss — exactly what makes win-rate and Elo
measurable and what the generalisation sweep varies (`N`).

### 8.2 Snapshot pool = δ-uniform self-play
`SnapshotPool` keeps up to 100 past snapshots (the whole 1.5 M-step run at a 10 k
interval) and samples an opponent **uniformly**. Training against a uniform mix of
*all* past selves (weak-early → strong-late), not just the latest, is δ-uniform
self-play (Heinrich & Silver; AlphaStar league): it avoids Red-Queen cycling and
makes "beating the pool" mean *dominating your own history*.

### 8.3 Opponent curriculum (learn to *hunt*, then to *compete*)
The fraction of episodes played against **random** opponents warms up at 1.0 and
decays to ~0.35. Pure self-play yields a *defensive* policy that can't eject a
passive random (0 % vs random); pure-random yields a *hunter* that only draws
against competent copies — the curriculum keeps both.

### 8.4 Rotation randomisation
Every reset rotates all spawn angles by a random `θ ∈ [0, 2π)`, decorrelating slot
from absolute position (§6) — this is what makes the balance demo hold at ≈ `1/N`.

### 8.5 Draw = loss
A timed-out draw (still alive, not the sole survivor) is charged `draw_penalty` so
stalling costs as much as losing. This destroys the cautious "run out the clock"
equilibrium — OpenAI's sumo trick (a draw scores −1000, like a loss). Set **equal
to** the death penalty, not larger, so there is no incentive to self-eliminate.
Payoff algebra: METHODOLOGY §3.3.

### 8.6 Agent-count curriculum + aggression boost
Training starts at `curriculum_start_agents = 2` and ramps to `num_agents`, so the
shove skill forms 1v1 (where committing has ~zero exposure) and is carried into the
melee. A potential-based **aggression boost** (`a`) is ramped up across the
transition and annealed back to 1 — scaling a potential leaves the optimum
unchanged, so it only speeds exploration (METHODOLOGY §3.5).

### 8.7 Elo
`EloRatingSystem` (K = 32) tracks the learner vs the pool via the standard logistic
update `R ← R + K·(S − E)` — a training-health signal (rising Elo = beating
progressively stronger snapshots).

---

## 9. PPO training
Stable-Baselines3 `PPO`, `MlpPolicy` (2×64), `lr = 3e-4`, `n_steps = 2048`,
`batch = 64`, `clip = 0.2`, `n_epochs = 10`, `gamma = 0.997`, single `DummyVecEnv`.
The high γ gives the terminal draw penalty an effective reach of `1/(1−γ) ≈ 333`
steps back into the mid-game where the commit-or-wait decision is made
(METHODOLOGY §3.4). A callback (a) records each match into `MetricsTracker`,
(b) snapshots into the pool every 10 k steps, (c) advances both curricula and the
aggression schedule. ~1.5 M steps ≈ 30–40 min on CPU.

---

## 10. Results (headline policy, `sumo_light`, 1.5 M steps)

| | n=2 | n=4 | n=6 | n=8 |
|---|---|---|---|---|
| **Active pushing** — self-play resolution, no storm (every resolution = a push-out) | 0.88 | **0.91** | 0.79 | 0.78 |
| **Dominance** — 1 trained vs n−1 random, edge = WIN−LOSS | +1.00 | **+0.98** | +0.85 | +0.56 |
| **Balance** — per-slot win share (ideal 1/n) | ~0.50 | ~0.25 | ≈ 1/6 | ≈ 1/8 |

n=8 is 2× the training count (zero-shot): pushing holds (0.78) and the learner is almost
never eliminated (dominance loss 0.01), but it cannot clear all seven randoms inside the time
limit, so the edge softens to +0.56.

Stable across every checkpoint 1.0 M–1.5 M. At n = 4 the policy wins 99 % of matches
against three untrained opponents; four copies of it resolve 91 % of games by an
actual push-out. Demo footage: `media/battle_royale_results.mp4` (captioned reel),
`media/sumo_light_n4.mp4`, `media/sumo_light_n6.mp4`.

---

## 11. Why the classic success target was replaced

The original target was ">60 % win-rate vs the pool". In a **symmetric** N-player
free-for-all this is provably unreachable: at a symmetric equilibrium all identical
policies are interchangeable, so each wins exactly `1/N` (0.25 at N = 4). You cannot
dominate copies of yourself. The three-behaviour framing measures what *is*
meaningful: **balance** (converged to a fair equilibrium), **dominance** (skill vs an
untrained baseline), and **active pushing** (the behaviour is real ejection, not a
timer).

---

## 12. The debugging chain (why every piece is load-bearing)

Each fix only revealed the next failure:

1. **Turtling** — survival reward beat winning. → rebalance; add **win bonus**.
2. **Suicide** — too-harsh time penalty made dying cheaper than living. → shrink it.
3. **Frictionless physics** — runaway drift-off. → **joint damping**.
4. **Reward too sparse** — a push-out never discovered. → **push shaping**.
5. **Positional overfit** — agent 0 wins ~80 % of identical matches. → **rotation randomisation**.
6. **Can't hunt a passive target** (0 % vs random). → **approach shaping** + **opponent curriculum**.
7. **Melee caution at n ≥ 3** — committing exposes you; all draws. → **agent-count curriculum** + **draw = loss + high γ**.
8. **Self-ejection ceiling** — agents overshoot the edge and die; dominance capped at −0.26 no matter the reward. → **light agents** (`d_brake < R_arena`) — the physics fix that unlocked all three targets at once.

---

## 13. Rendering
`main.py` drives an episode and records video. The collision cylinder is invisible
(alpha 0); a **visual-only** humanoid (zero mass, no collision) is drawn over it, so
physics is unchanged. Each agent keeps a fixed colour (`agent_0` = red) so the
dominance demo (`--dominance`: only red runs the policy) reads at a glance. A framed
tracking camera and a 720p offscreen buffer produce the final clips.
