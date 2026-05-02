from battle_royale.domain.entities.agent import Agent

_ELIMINATION_REWARD = 1.0
_DEATH_PENALTY = -1.0
_SURVIVAL_REWARD = 0.01


class RewardCalculator:
	@staticmethod
	def compute(
		prev_agents: dict[str, Agent],
		curr_agents: dict[str, Agent],
		agent_id: str,
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
			reward += _DEATH_PENALTY
		else:
			reward += _SURVIVAL_REWARD

		return reward
