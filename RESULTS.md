# Results

## Headline

A **single self-play policy** (`config/sumo_light.yaml`, `runs/sumo_light`) demonstrates
all three target behaviours in an N-agent free-for-all, at **n=4 and n=6** — the hard
multi-agent regime, not just 1v1:

| Target | n=2 | n=4 | n=6 | n=8 |
|---|---|---|---|---|
| **Active pushing** — self-play resolution with **no storm**, so every resolution is a genuine push-out | 0.88 | **0.91** | 0.79 | 0.78 |
| **Dominance** — 1 trained vs n−1 random, edge = WIN−LOSS | **+1.00** | **+0.98** | **+0.85** | +0.56 |
| **Balance** — per-slot win share among N identical agents (ideal 1/n) | ~0.50 | ~0.25 | ≈[.14 .15 .09 .09 .16 .16] | ≈0.13 |

At n=4 the policy wins 99% of matches against three untrained opponents (LOSS 0.01), and
when four copies of it play each other 91% of games resolve by an actual push-out with a
roughly uniform winner distribution. The result is stable across every checkpoint from
1.0M–1.5M steps (n=4 pushing 0.87–0.98, n=4 dominance +0.80 to +1.00).

**Generalisation to n=8** (2× the training count, zero-shot; 100 deterministic episodes):
active pushing holds at **0.78** and the trained agent is **almost never eliminated
(LOSS 0.01)** — but at this density it cannot always clear all seven random opponents inside
the time limit, so the dominance edge softens to **+0.56** (WIN 0.57). Balance also spreads
more at n=8 (per-slot ≈ [.10 .04 .12 .14 .02 .10 .14 .12], ideal 0.125): the residual
observation-tie bias (see METHODOLOGY §3.7) compounds with agent count. An independent
re-measurement corroborates the headline n=4/n=6 figures within sampling noise.

## The key finding: a physics flaw, not a reward flaw

Every earlier attempt to get active pushing at n≥3 collapsed into cautious draws or
self-elimination, and no reward change fixed it. The root cause was in the **physics**:

- The collision cylinder used MuJoCo's default density (1000 kg/m³) → **~7 kg** agents.
- With joint damping 4 and terminal velocity ~3.5 m/s, the braking distance
  (`v · m / c`) is **~5.5 m** — but the arena is only ~1.2 m across.
- So an agent that committed to real speed **physically could not stop inside the ring.**
  That single fact forced *both* symptoms we kept seeing: self-ejection (overshooting the
  edge) and caution (the only way not to fly out was to move slowly and never commit).

Lowering the cylinder density to **150 kg/m³** (~1.1 kg) cuts the braking distance to
**~0.96 m** — it fits the ring — while keeping terminal velocity and making shoves
displace opponents *more*. This is exposed as `arena.agent_density` (default 1000
preserves the original physics).

## The winning recipe

1. **Light agents** (`agent_density: 150`) — can brake inside the ring; the physics fix.
2. **Draw = loss** (`training.draw_penalty: 1.0`) — a timed-out draw is charged the same
   as a death, so the cautious "run out the clock" strategy stops paying. This mirrors
   OpenAI's *Emergent Complexity via Multi-Agent Competition* sumo, where a draw scores
   −1000 exactly like a loss. Kept equal to (not worse than) the death penalty so there
   is no incentive to self-eliminate.
3. **No storm** (`min_radius_frac: 1.0`) — with no closing boundary to resolve games for
   free, pushing an opponent out is the *only* way to win, so wins are genuine push-outs.
4. **Longer horizon reach** (`ppo.gamma: 0.997`) — so the terminal draw penalty propagates
   back into the mid-game where the caution happens.
5. **Agent-count curriculum** (`curriculum_start_agents: 2`) + an annealed aggression
   boost — the shove behaviour forms 1v1 (no melee risk) and is carried into the melee.

## Experiment log (summary)

| Run | Change | n=4 pushing | n=4 dominance |
|---|---|---|---|
| direct n=4 sumo | train straight at n=4 | 0.07 | — |
| curriculum3 | n=2→4 curriculum + aggression boost, no storm | 0.64 | −0.33 |
| sumo_arena | + smaller arena + storm | ~0 (storm ejects) | +0.08 |
| sumo_openai | + draw=loss (no storm) | 0.26 | −0.26 (n=2 **+0.90**) |
| sumo_openai_r10 | + shrink arena to 1.0 | 0.42 | −0.33 |
| **sumo_light** | **+ light agents (density 150)** | **0.91** | **+0.98** |

The progression shows the two independent levers that finally combined: OpenAI's draw=loss
incentive (which alone nailed 1v1 dominance, +0.90) and the mass/braking-distance physics
fix (which removed the self-ejection ceiling that capped every n≥3 result).
