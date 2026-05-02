import numpy as np
import mujoco
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena
from battle_royale.domain.services.elimination import EliminationService
from battle_royale.infrastructure.physics.xml_builder import XMLBuilder
from battle_royale.infrastructure.config.yaml_loader import Config


class MuJoCoEnvironment:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._num_agents: int = 0
        self._alive: dict[str, bool] = {}
        self._arena: Arena = Arena(radius=config.arena.radius)

    def reset(self, num_agents: int) -> dict[str, Agent]:
        self._num_agents = num_agents
        self._arena = Arena(radius=self._config.arena.radius)
        xml = XMLBuilder.build(
            num_agents=num_agents,
            arena_radius=self._config.arena.radius,
            max_force=self._config.training.max_force,
        )
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._alive = {f"agent_{i}": True for i in range(num_agents)}
        return self._extract_agents()

    def step(self, actions: dict[str, np.ndarray]) -> tuple[dict, dict, dict, dict]:
        assert self._model is not None and self._data is not None

        prev_agents = self._extract_agents()

        # Apply actions scaled by motor gear (already baked into MJCF ctrlrange * gear)
        for i in range(self._num_agents):
            agent_id = f"agent_{i}"
            if not self._alive.get(agent_id, False):
                # Freeze eliminated agents
                self._data.ctrl[2 * i] = 0.0
                self._data.ctrl[2 * i + 1] = 0.0
                self._data.qvel[2 * i] = 0.0
                self._data.qvel[2 * i + 1] = 0.0
            else:
                action = actions.get(agent_id, np.zeros(2))
                self._data.ctrl[2 * i] = float(np.clip(action[0], -1.0, 1.0))
                self._data.ctrl[2 * i + 1] = float(np.clip(action[1], -1.0, 1.0))

        mujoco.mj_step(self._model, self._data)

        curr_agents_raw = self._extract_agents_raw()

        # Run elimination
        for i in range(self._num_agents):
            agent_id = f"agent_{i}"
            if not self._alive[agent_id]:
                continue
            raw_agent = curr_agents_raw[agent_id]
            temp_agent = Agent(
                id=agent_id,
                position=raw_agent["position"],
                velocity=raw_agent["velocity"],
                alive=True,
            )
            if EliminationService.is_eliminated(temp_agent, self._arena):
                self._alive[agent_id] = False
                # Freeze in place
                self._data.qvel[2 * i] = 0.0
                self._data.qvel[2 * i + 1] = 0.0
                self._data.ctrl[2 * i] = 0.0
                self._data.ctrl[2 * i + 1] = 0.0

        curr_agents = self._extract_agents()

        from battle_royale.domain.services.reward import RewardCalculator

        rewards = {
            aid: RewardCalculator.compute(prev_agents, curr_agents, aid)
            for aid in curr_agents
        }
        terminations = {aid: not agent.alive for aid, agent in curr_agents.items()}
        truncations = {aid: False for aid in curr_agents}

        return curr_agents, rewards, terminations, truncations

    def get_agents(self) -> list[Agent]:
        return list(self._extract_agents().values())

    def _body_id(self, i: int) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, f"agent_{i}")

    def _extract_agents_raw(self) -> dict[str, dict]:
        result = {}
        for i in range(self._num_agents):
            agent_id = f"agent_{i}"
            bid = self._body_id(i)
            result[agent_id] = {
                "position": np.array(self._data.xpos[bid, :2], dtype=np.float64),
                "velocity": np.array(
                    self._data.qvel[2 * i : 2 * i + 2], dtype=np.float64
                ),
            }
        return result

    def _extract_agents(self) -> dict[str, Agent]:
        agents = {}
        for i in range(self._num_agents):
            agent_id = f"agent_{i}"
            bid = self._body_id(i)
            agents[agent_id] = Agent(
                id=agent_id,
                position=np.array(self._data.xpos[bid, :2], dtype=np.float64),
                velocity=np.array(self._data.qvel[2 * i : 2 * i + 2], dtype=np.float64),
                alive=self._alive.get(agent_id, True),
            )
        return agents


# Backward-compatible module-level functions for main.py
def load_model() -> mujoco.MjModel:
    from battle_royale.infrastructure.physics.xml_builder import build

    return mujoco.MjModel.from_xml_string(build())


def load_data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)


def render(model: mujoco.MjModel, data: mujoco.MjData, output_path: str) -> None:
    import mediapy

    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()
    mediapy.write_image(output_path, frame)
