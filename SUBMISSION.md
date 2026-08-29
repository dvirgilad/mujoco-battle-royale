# Submission guide — Multi-Agent Battle Royale

Everything in this project, and where to find it. Start here.

## The result in one line

A **single self-play PPO policy** makes N cylindrical agents compete in a circular
sumo arena and demonstrates **three behaviours at once** in the hard melee regime
(n = 4 and n = 6, and generalising zero-shot to n = 8): they **balance** (each of N wins
≈ 1/N), one trained agent **dominates** N−1 untrained ones (+0.98 edge at n=4), and they win
by **actively pushing** opponents out (91% of n=4 self-play games resolve by a real push-out).
At n=8 — 2× the training count — pushing still holds (0.78) and the learner is almost never
eliminated (loss 0.01), though the dominance edge softens to +0.56.
The unlock was a **physics** fix, not a reward fix — see below.

## Deliverables

| What | File | Read it for |
|---|---|---|
| **Presentation** | [`docs/battle-royale-presentation.pptx`](docs/battle-royale-presentation.pptx) | 12-slide overview of the whole project |
| **Design & results report** | [`docs/battle-royale-design-doc-final.docx`](docs/battle-royale-design-doc-final.docx) | the polished formal design document |
| **Results video** | [`media/battle_royale_results.mp4`](media/battle_royale_results.mp4) | ~80-s captioned reel: pushing (n=4), generalisation (n=6, n=8), dominance (n=4, n=8) + results |
| **Methodology (the maths)** | [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | every experiment, dead end, and the equations behind each fix |
| **System design (the logic)** | [`docs/DESIGN.md`](docs/DESIGN.md) | architecture, physics, reward, self-play — what & why, per component |
| **Runbook** | [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | install, train, evaluate, render — copy-paste commands |
| **Results summary** | [`RESULTS.md`](RESULTS.md) | the headline table + the physics root-cause, one page |
| **Project README** | [`README.md`](README.md) | quick orientation + architecture |
| **Raw demo clips** | `media/sumo_light_n4.mp4` · `_n6.mp4` · `_n8.mp4`; `media/dominance_n4.mp4` · `dominance_n8.mp4` | uncaptioned self-play push-outs (n=4/6/8) and dominance renders (1 red trained vs the field, n=4/8) |

## The three claims and how to verify each

| Claim | Evidence | Reproduce |
|---|---|---|
| **Active pushing** | 0.91 self-play resolution at n=4 with **no storm**, so every resolution is a genuine ejection (0.78 at n=8) | `evaluate --sweep` on the `sumo_light` checkpoint |
| **Dominance** | +0.98 edge (WIN 0.99) vs 3 random opponents at n=4 (+0.56 at n=8, loss 0.01) | `main.py --dominance` (red = trained) / evaluator |
| **Balance** | per-slot win share ≈ 1/N (small, explained residual bias; spreads more at n=8) | N identical copies self-play sweep |

## The one thing to remember

Every reward change failed to produce melee pushing until we found the real cause
in the **physics**: at the default agent mass the braking distance from top speed
is **~5.5 m**, but the arena is only **~1.2 m** — an agent that commits to any real
speed *cannot stop inside the ring*, which forced both caution and self-ejection.
Lightening the agents (density 1000 → 150) cuts braking to ~0.96 m, and that single
change turned a −0.26 dominance ceiling into +0.98. The full derivation and the
honest negative results (three balance-perfection attempts that all collapsed the
aggression) are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Reproduce the headline policy

```bash
poetry install
python -m battle_royale.interfaces.cli.train    --config config/sumo_light.yaml --run-dir runs/sumo_light --logger stdout
python -m battle_royale.interfaces.cli.evaluate  --checkpoint runs/sumo_light/snapshots/snapshot_001500000.zip --run-dir runs/sumo_light --sweep
```

Config: `config/sumo_light.yaml`. Tests: `poetry run pytest tests/` (134 passing, ≥80% coverage gate).
