import math

_AGENT_COLORS = [
    (1, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (0.5, 0.5, 0),
    (0, 0.5, 0.5),
]

_CYLINDER_RADIUS = 0.15
_CYLINDER_HALF_HEIGHT = 0.05
_SPAWN_RADIUS_FRACTION = 0.6


class XMLBuilder:
    @staticmethod
    def build(num_agents: int, arena_radius: float, max_force: float) -> str:
        spawn_r = _SPAWN_RADIUS_FRACTION * arena_radius
        bodies_xml = ""
        motors_xml = ""

        for i in range(num_agents):
            angle = (i * 2 * math.pi) / num_agents
            x = spawn_r * math.cos(angle)
            y = spawn_r * math.sin(angle)
            r, g, b = _AGENT_COLORS[i % len(_AGENT_COLORS)]
            bodies_xml += f"""
    <body name="agent_{i}" pos="{x:.6f} {y:.6f} {_CYLINDER_HALF_HEIGHT}">
      <joint name="agent_{i}_x" type="slide" axis="1 0 0" limited="false"/>
      <joint name="agent_{i}_y" type="slide" axis="0 1 0" limited="false"/>
      <geom type="cylinder" size="{_CYLINDER_RADIUS} {_CYLINDER_HALF_HEIGHT}" rgba="{r} {g} {b} 1"/>
    </body>"""
            motors_xml += f"""
    <motor name="agent_{i}_motor_x" joint="agent_{i}_x" gear="{max_force}" ctrlrange="-1 1"/>
    <motor name="agent_{i}_motor_y" joint="agent_{i}_y" gear="{max_force}" ctrlrange="-1 1"/>"""

        return f"""<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <geom type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>{bodies_xml}
  </worldbody>
  <actuator>{motors_xml}
  </actuator>
</mujoco>"""


# Backward-compatible module-level function for main.py
def build() -> str:
    return XMLBuilder.build(num_agents=1, arena_radius=3.0, max_force=10.0)
