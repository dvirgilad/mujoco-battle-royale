import numpy as np

from battle_royale.domain.entities.agent import Agent

_ELIMINATION_REWARD = 1.0
_DEATH_PENALTY = -1.0
# Tiny positive so staying alive always beats dying (removes any incentive to
# self-eliminate). Kept small so passive turtling to the step cap accrues only a
# negligible return compared with the win bonus below.
_SURVIVAL_REWARD = 0.001
# Small per-step cost so net-per-idle-step is slightly negative (survival +
# time = -0.002). This makes turtling to the 400-step cap net ~-0.8 -- worse
# than doing nothing but still comfortably above the death penalty (-1), so it
# breaks the passive-turtle local optimum WITHOUT recreating a suicide incentive
# (an earlier -0.01 penalty made idling worse than dying and the policy learned
# to self-eliminate). Winning (+win bonus) dominates either way.
_TIME_PENALTY = -0.003
# Awarded once, when the agent is the sole survivor (i.e. it won the match).
# This is what makes winning dominate turtling: turtling to a 400-step cap nets
# ~+0.4, whereas winning nets eliminations (+1 each) plus this bonus. Without an
# explicit win signal the policy either turtles (if survival is large) or
# self-eliminates (if survival is negative) — see docs/next-steps.md.
_WIN_BONUS = 10.0
# Dense shaping: reward per unit an opponent's distance-from-centre increases
# (i.e. per unit it is driven toward the edge). The elimination/win rewards are
# too sparse for random exploration to ever stumble on a full push-out, so the
# policy gets stuck turtling. This potential-based term (potential = opponent
# distance-from-centre) gives a smooth gradient toward the real objective while
# leaving the optimal policy unchanged. Only applies to opponents alive both
# before and after the step, so it never double-counts an elimination.
_PUSH_COEF = 1.0
# Dense shaping: reward per unit the agent closes the gap to its nearest
# opponent. Pure self-play produces a defensive policy that never learns to
# *hunt* -- opponents come to it -- so it fails to eject passive targets. This
# term provides a gradient toward engagement from anywhere in the arena. It does
# not fight _PUSH_COEF: while the agent stays in contact and shoves, the gap
# stays ~constant (this term ~0); it is only large during the initial approach,
# and goes negative if the agent disengages (which we want to discourage).
_APPROACH_COEF = 0.4
# Dense self-preservation penalty, active only in the outer ring (beyond
# _EDGE_SAFE_FRAC of the arena radius). The agent kept overshooting its own edge
# and self-eliminating while chasing; a terminal death penalty is too sparse to
# teach braking, so this gives a gradient that grows as the agent nears the
# boundary, pushing it to slow down / turn back. Confined to the outer ring so
# it doesn't discourage central maneuvering or driving an opponent to the edge.
_EDGE_SAFE_FRAC = 0.92
_EDGE_PENALTY_COEF = 1.0


class RewardCalculator:
    @staticmethod
    def compute(
        prev_agents: dict[str, Agent],
        curr_agents: dict[str, Agent],
        agent_id: str,
        arena_radius: float = 3.0,
    ) -> float:
        prev_self = prev_agents[agent_id]
        curr_self = curr_agents[agent_id]

        if not prev_self.alive:
            return 0.0

        reward = 0.0

        for aid, prev_agent in prev_agents.items():
            if aid == agent_id:
                continue
            if prev_agent.alive and not curr_agents[aid].alive:
                reward += _ELIMINATION_REWARD

        if not curr_self.alive:
            return reward + _DEATH_PENALTY

        reward += _SURVIVAL_REWARD + _TIME_PENALTY

        # Dense self-preservation: penalise being in the outer ring so the agent
        # learns to brake before its own edge instead of overshooting to death.
        safe_radius = _EDGE_SAFE_FRAC * arena_radius
        own_dist = float(np.linalg.norm(curr_self.position))
        if own_dist > safe_radius:
            reward += _EDGE_PENALTY_COEF * (safe_radius - own_dist)

        # Opponents alive both before and after this step (valid deltas).
        shared = [
            aid
            for aid, prev_agent in prev_agents.items()
            if aid != agent_id and prev_agent.alive and curr_agents[aid].alive
        ]

        # Dense shaping: reward pushing still-alive opponents outward.
        for aid in shared:
            prev_d = float(np.linalg.norm(prev_agents[aid].position))
            curr_d = float(np.linalg.norm(curr_agents[aid].position))
            reward += _PUSH_COEF * (curr_d - prev_d)

        # Dense shaping: reward closing the gap to the nearest opponent (hunting).
        if shared:
            prev_gap = min(
                float(np.linalg.norm(prev_agents[aid].position - prev_self.position))
                for aid in shared
            )
            curr_gap = min(
                float(np.linalg.norm(curr_agents[aid].position - curr_self.position))
                for aid in shared
            )
            reward += _APPROACH_COEF * (prev_gap - curr_gap)

        # Sole-survivor win bonus (only meaningful when opponents exist).
        others = [aid for aid in curr_agents if aid != agent_id]
        if others and all(not curr_agents[aid].alive for aid in others):
            reward += _WIN_BONUS

        return reward
