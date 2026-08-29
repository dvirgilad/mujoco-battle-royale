# Methodology: getting agents to actively push at n≥4

This document is the full research narrative behind `sumo_light` — the goal, every
approach we tried, what worked, what didn't, and **why**, with the mathematics that
explains each outcome. It is deliberately honest about the dead ends, because the dead
ends are what localised the real bottleneck.

---

## 1. The goal

Three behaviours in an N-agent circular sumo free-for-all, ideally from **one** policy,
and crucially in the **melee** regime (n ≥ 4), not just 1v1:

1. **Balance** — N identical trained agents each win ≈ 1/N of the time.
2. **Dominance** — 1 trained agent beats N−1 untrained/random agents.
3. **Active pushing** — agents win by *shoving opponents out*, not by outlasting a
   shrinking "storm" boundary.

(1) and (2) are the CS-submission targets; (3) is the behaviour that makes the result
interesting rather than a survival timer.

---

## 2. The result

`config/sumo_light.yaml` → `runs/sumo_light`:

| | n=2 | n=4 | n=6 | n=8 |
|---|---|---|---|---|
| Active pushing (self-play resolution, **no storm**) | 0.88 | **0.91** | 0.79 | 0.78 |
| Dominance edge (WIN−LOSS vs random) | +1.00 | **+0.98** | +0.85 | +0.56 |

Stable across all checkpoints 1.0M–1.5M. Every self-play resolution is a genuine
push-out because there is no storm to eliminate anyone for free.

**Zero-shot to n=8** (2× the training count; 100 deterministic episodes). Active pushing
holds at 0.78, and the learner is almost never eliminated (dominance LOSS 0.01) — but it can
no longer clear all seven random opponents inside the 600-step budget, so the dominance edge
softens to +0.56 (WIN 0.57). Balance also degrades (per-slot ≈ [.10 .04 .12 .14 .02 .10 .14 .12]
vs ideal 0.125): the observation-tie bias of §3.7 compounds with agent count. So the policy
generalises in *behaviour* (it still pushes and is never beaten by randoms) but not in raw
throughput (clearing a denser field needs more time than the episode allows).

---

## 3. The mathematics

### 3.1 The arena physics — the hidden bottleneck

Each agent is a cylinder on two orthogonal slide joints. Per axis the dynamics are

    m·v̇ = F − c·v

where `m` = cylinder mass, `F` = motor force (= `max_force`·action, |action|≤1),
`c` = joint damping. Two quantities follow:

- **Terminal velocity** `v_∞ = F/c` (set v̇=0). Independent of mass.
- **Velocity time-constant** `τ = m/c`. After cutting thrust, `v(t) = v₀·e^(−t/τ)`, so the
  **passive braking distance** from terminal velocity is

      d_brake = ∫₀^∞ v(t) dt = v₀·τ = (F/c)·(m/c) = F·m / c²

The masses in play: the collision cylinder (r=0.15, h=0.10 → volume `V = π r² h ≈ 7.07e-3 m³`)
at MuJoCo's default density ρ=1000 kg/m³ gives `m = ρV ≈ 7.07 kg`. With `c=4`, `F=14`:

      v_∞ = 14/4 = 3.5 m/s
      d_brake = 14·7.07 / 4² ≈ 6.2 m   (measured ≈ 5.5 m with reverse thrust available)

**But the arena radius is ~1.2 m.** So `d_brake ≫ R_arena`: an agent that accelerates to
any real speed **physically cannot stop inside the ring.** This single inequality explains
*both* failure modes we chased for a dozen experiments:

- **Self-ejection** — chasing an opponent, the agent overshoots the boundary and dies.
- **Caution** — the only way *not* to fly out is to keep `v` small, i.e. never commit to a
  push. No reward change can defeat this, because it is a hard dynamical constraint.

Since `d_brake ∝ m ∝ ρ`, lowering the cylinder density is the direct fix. We expose
`arena.agent_density`; at ρ=150 kg/m³ (`m ≈ 1.06 kg`):

      d_brake ≈ 14·1.06 / 4² ≈ 0.93 m   (measured 0.96 m)  <  R_arena ✓

`v_∞` is unchanged (it doesn't depend on `m`), and a lighter opponent is displaced *more*
per unit impulse, so pushing gets *easier* at the same time. This is the load-bearing
change: it turns "cannot stop in the ring" into "can stop in the ring."

### 3.2 The melee-risk equilibrium — why n ≥ 3 is hard

In 1v1 sumo, pushing is unambiguously good: shove the only opponent out and you win. In an
n-player free-for-all it is not. If you commit to a shove on opponent B, for the duration
of that commitment you present a predictable, high-momentum target to opponents C, D, … who
can knock *you* out. Let committing have win-probability `p` and self-elimination
probability `q`; the exposure term `q` grows with the number of other agents. The
symmetric best response is therefore to **wait** and let someone else over-commit first — a
cautious Nash equilibrium in which nobody pushes and games go to a draw. This is exactly
the "0.07 resolution, all draws" we measured when training directly at n=4.

Two things break this equilibrium (see §4): starting the curriculum at n=2 so the shove
skill forms with `q≈0` and is then carried into the melee; and making a draw *cost*
something (§3.3) so waiting is no longer free.

### 3.3 Draw = loss — the payoff algebra

Give terminal payoffs: win `+W`, death `−L`, draw `−D`. Consider "wait" (→ draw, payoff
`−D`) vs "commit" (win w.p. `p`, die w.p. `q`, payoff `p·W − q·L`). Commit is preferred iff

      p·W − q·L  >  −D

- **Draws free (`D = 0`, our original reward):** condition is `p·W > q·L`. In a melee `q` is
  large and `p` small, so it fails — **waiting wins**, which is the caution we observed.
- **Draw = loss (`D = L`):** condition becomes `p·W > q·L − L = (q−1)L`. Since `q ≤ 1`, the
  right side is `≤ 0 < p·W`, so it holds for **any** positive win chance — **committing
  always weakly dominates waiting.**

This is OpenAI's *Emergent Complexity via Multi-Agent Competition* trick: their sumo scores
a timed-out draw at −1000, identical to a loss. We set `draw_penalty = L` (equal to the
death penalty, not larger) so committing is forced *without* making a draw worse than
dying — which would revive an incentive to self-eliminate.

### 3.4 Discounting — why the draw penalty needs a high γ

A terminal penalty at step `T` is worth `γ^T` at the episode start; more usefully, TD
propagation gives it an effective reach of `H = 1/(1−γ)` steps back from the terminus.

      γ = 0.99  → H = 100 steps
      γ = 0.997 → H = 333 steps

Our episodes are ~600 steps and the caution happens throughout, not just in the final 100.
At γ=0.99 the draw penalty is invisible until the very end (`γ^600 ≈ 2.4e-3`); at γ=0.997 it
reaches back ~333 steps into the mid-game where the decision to commit-or-wait is actually
made. Hence `ppo.gamma = 0.997`.

### 3.5 Potential-based shaping is optimum-invariant

The dense push/approach terms are potential-based: `F = γΦ(s') − Φ(s)` with
`Φ_push = −(opponent distance from centre)` and `Φ_approach = −(gap to nearest opponent)`.
By Ng, Harada & Russell (1999), adding such an `F` leaves the optimal policy unchanged, and
scaling `Φ → αΦ` still yields a valid potential — so the **aggression boost** (`α` ramped up
during the melee transition, annealed back to 1) changes only *exploration/gradient
strength*, never the objective's optimum. That is why we can crank it up to bootstrap
pushing through the melee-risk transition and then remove it safely.

### 3.6 Balance = symmetric-game exchangeability

With N agents sharing one policy in a symmetric arena (spawns on a regular n-gon, randomised
whole-arena rotation each episode), the agents are exchangeable, so the win distribution is
uniform in expectation: each slot wins `1/N`. Deviations from `1/N` therefore measure
residual symmetry-breaking (positional overfitting or correlated determinism), which is why
rotation randomisation was necessary to pull agent_0's win share down from ~80% to ~1/N.

### 3.7 The residual balance bias — an observation tie-break

`sumo_light` still showed a mild per-slot bias at n=4 (win shares ≈ [.20 .19 .32 .29],
persistent at N=300 and under stochastic actions, so *not* sampling noise). The cause is a
subtle interaction between the spawn geometry and the observation:

- On a regular n-gon each agent's two *adjacent* neighbours are **exactly equidistant**
  (e.g. at n=4 both are 1.018 units away, measured).
- The observation lists the 3 nearest neighbours **sorted by distance**. With an exact tie,
  Python's stable sort keeps the two adjacent neighbours in **agent-index order**.
- So "neighbour #1" in the observation vector is the *counter-clockwise* neighbour for some
  slots and the *clockwise* neighbour for others. The policy treats that first slot
  specially, so slots inherit a fixed CW/CCW role and a systematic win bias.

We tried three fixes for this bias, and **all three collapsed the aggressive behaviour** —
a result more interesting than the bias itself:

| Fix | Balance | Dominance / pushing |
|---|---|---|
| `spawn_jitter` (angular+radial) | evened (spread 0.19→0.04) | dominance +0.98→+0.02 |
| `spawn_jitter` (angular only) | evened | dominance +0.98→−0.12, pushing 0.91→0.24 |
| `shuffle_neighbors` (observation only) | — | **total passivity**: resolution 0.00 at every n |

The common cause is **not** the physics (the observation-shuffle run leaves the dynamics
identical to `sumo_light` and still collapsed). It is **training-time stochasticity**. The
draw-penalty solution (§3.3) is a knife-edge: the policy must *learn to clear opponents* to
escape the draw penalty, and clearing is hard. `sumo_light` barely learns it in a clean,
deterministic environment; adding *any* noise — perturbed spawns, or a shuffled neighbour
order — makes clearing too hard to master in-budget, so the policy retreats to the safe
passive draw-equilibrium (survive, eat the penalty, never commit). The aggressive optimum is
**fragile to training noise**.

Consequently the balance bias is left as a documented, fully-explained artifact rather than
"fixed": `sumo_light`'s per-slot shares at n=4 are ≈ [.20 .19 .32 .29] — *approximately* 1/N
(within ±7% of ideal), acceptable for the balance demonstration, and not worth sacrificing
the +0.98 dominance and 0.91 pushing that the primary targets require. The `spawn_jitter` and
`shuffle_neighbors` knobs remain in the code (default off) as documented, reproducible
negative results. Removing the bias without the fragility cost would need a genuinely
permutation/rotation-*equivariant* policy architecture — future work, not a config tweak.

---

## 4. What we tried — the full log

Each row is a training run; the "why" is the mechanism above that explains it.

| # | Config | Idea | n=4 pushing | n=4 dominance | Verdict / why |
|---|---|---|---|---|---|
| 0 | direct n=4 | train straight at n=4 | 0.07 | — | melee-risk equilibrium (§3.2): all draws |
| 1 | curriculum3 | n=2→4 curriculum + annealed aggression boost, **no storm** | **0.64** | −0.33 | curriculum carries the shove into the melee; but can't clear passive randoms |
| 2 | (variant) | + more random-opponent exposure | 0.06 | +0.00 (all draws) | **backfired**: passive opponents teach turtling — pushing and robustness are antagonistic under a free draw |
| 3 | sumo_arena | + smaller arena + **storm** | ~0 (storm ejects) | +0.08 | storm forces resolution → dominance up, but removes the *incentive* to push (grace-phase push-outs = 0) |
| 4 | sumo_openai | **draw = loss** (§3.3), no storm | 0.26 | −0.26; **n=2 +0.90** | draw=loss nails the 1v1 case exactly like OpenAI; n≥3 still limited by self-ejection |
| 5 | sumo_openai_r10 | + shrink arena to 1.0 | 0.42 | −0.33 | smaller arena helps *pushing* but adds chaos vs random → hurts dominance; does **not** fix self-ejection |
| 6 | **sumo_light** | **+ light agents** (ρ=150, §3.1) | **0.91** | **+0.98** | braking distance < arena → self-ejection gone; pushing + dominance + balance all at once |
| 7 | sumo_light_v2 | + spawn jitter (ang+radial) to even balance (§3.7) | 0.97 | +0.02 | balance evened (spread→0.04) but dominance collapsed — perturbed dynamics → passive |
| 8 | sumo_light_v3 | + spawn jitter (angular only) | 0.24 | −0.12 | still collapses: the perturbation, not the radial part, is the problem |
| 9 | sumo_light_v4 | + observation neighbour-shuffle (no physics change) | 0.00 | +0.04 | total passivity — training-time noise alone breaks the fragile aggressive optimum |

**What worked:**
- **Agent-count curriculum** (n=2→4): forms the shove at `q≈0`, carries it into the melee.
- **Draw = loss + high γ**: removes the free-draw refuge that made caution rational (§3.3–3.4).
- **Light agents**: the physics fix that made braking possible in the ring (§3.1) — the
  single change that converted a −0.33 dominance ceiling into +0.98.
- **No storm**: makes pushing the *only* win path, so wins are genuine push-outs.

**What didn't, and why:**
- **More random opponents** (#2): passive opponents make turtling the safe reward-maximiser;
  antagonistic to pushing while draws are free.
- **Storm for the melee** (#3): resolves games but *replaces* pushing rather than causing it,
  and rewards non-contact wins — the opposite of the goal.
- **Just shrinking the arena** (#5): crowds the dynamics and increases random-knockout losses
  without touching the braking-distance root cause.
- **Reward tuning in general, before the physics fix**: bounded above by a dynamical
  constraint (`d_brake ≫ R`) that no reward can override.

---

## 5. The winning recipe (`config/sumo_light.yaml`)

```yaml
arena:   { radius: 1.2, damping: 4.0, agent_density: 150, min_radius_frac: 1.0 }  # light, no storm
training:{ curriculum_start_agents: 2, draw_penalty: 1.0, episode_max_steps: 600, max_force: 14 }
ppo:     { gamma: 0.997 }
```

- `agent_density: 150` — braking distance ≈ 0.96 m < arena (§3.1)
- `draw_penalty: 1.0` — draw = loss, equal to the death penalty (§3.3)
- `gamma: 0.997` — terminal penalty reaches the mid-game (§3.4)
- `min_radius_frac: 1.0` — no storm; pushing is the only way to win
- `curriculum_start_agents: 2` — form the shove 1v1, carry it into the melee (§3.2)

Demo videos: `media/sumo_light_n4.mp4`, `media/sumo_light_n6.mp4`.
