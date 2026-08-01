import math

_AGENT_COLORS = [
    (0.90, 0.20, 0.20),
    (0.20, 0.45, 0.90),
    (0.20, 0.75, 0.35),
    (0.95, 0.80, 0.20),
    (0.80, 0.30, 0.85),
    (0.20, 0.80, 0.80),
    (0.95, 0.55, 0.15),
    (0.55, 0.55, 0.60),
]

_CYLINDER_RADIUS = 0.15
_CYLINDER_HALF_HEIGHT = 0.05
_SPAWN_RADIUS_FRACTION = 0.6
# Velocity damping on the slide joints. Without it the arena is frictionless, so
# any net motor force integrates into unbounded velocity and agents slide off the
# edge uncontrollably (the task becomes unlearnable). Damping gives a bounded
# terminal velocity (~F/damping) and lets an agent stop by zeroing its action,
# which is what makes "stay in, push others out" a learnable sumo skill.
_JOINT_DAMPING = 8.0
# Skin tone for the visual "head" geom (same for all agents).
_SKIN = (0.96, 0.80, 0.66)


def _humanoid_geoms(r: float, g: float, b: float) -> str:
    """Visual-only humanoid, built from primitives above the collision base.

    Every geom here has ``density="0"`` (adds no mass/inertia) and
    ``contype="0" conaffinity="0"`` (no collisions), so the physics is identical
    to a bare cylinder -- these geoms only change how the agent is *drawn*. The
    real collision/mass shape is the (invisible) cylinder emitted separately.
    Coordinates are body-relative; the body origin sits at world z=0.05.
    """
    body = f'rgba="{r} {g} {b} 1" contype="0" conaffinity="0" density="0"'
    skin = f'rgba="{_SKIN[0]} {_SKIN[1]} {_SKIN[2]} 1" contype="0" conaffinity="0" density="0"'
    return f"""
      <geom type="capsule" fromto="0.055 0 -0.05 0.055 0 0.14" size="0.035" {body}/>
      <geom type="capsule" fromto="-0.055 0 -0.05 -0.055 0 0.14" size="0.035" {body}/>
      <geom type="capsule" fromto="0 0 0.14 0 0 0.34" size="0.1" {body}/>
      <geom type="capsule" fromto="0 0 0.32 0.17 0 0.2" size="0.03" {body}/>
      <geom type="capsule" fromto="0 0 0.32 -0.17 0 0.2" size="0.03" {body}/>
      <geom type="sphere" pos="0 0 0.46" size="0.09" {skin}/>"""


class XMLBuilder:
    @staticmethod
    def build(
        num_agents: int,
        arena_radius: float,
        max_force: float,
        rotation: float = 0.0,
        damping: float = _JOINT_DAMPING,
    ) -> str:
        # ``rotation`` offsets every spawn angle by the same amount. Randomising
        # it per episode decorrelates each slot from a fixed absolute position,
        # so the policy cannot memorise (e.g.) "agent_0 starts at (spawn_r, 0)"
        # and must become rotation-invariant. Without it the learned policy is
        # wildly asymmetric across slots (agent_0 wins ~80% of an all-identical
        # free-for-all instead of ~1/N).
        spawn_r = _SPAWN_RADIUS_FRACTION * arena_radius
        bodies_xml = ""
        motors_xml = ""

        for i in range(num_agents):
            angle = (i * 2 * math.pi) / num_agents + rotation
            x = spawn_r * math.cos(angle)
            y = spawn_r * math.sin(angle)
            r, g, b = _AGENT_COLORS[i % len(_AGENT_COLORS)]
            # The cylinder is the real collision + mass shape but drawn
            # invisibly (alpha 0); the humanoid geoms are visual only.
            bodies_xml += f"""
    <body name="agent_{i}" pos="{x:.6f} {y:.6f} {_CYLINDER_HALF_HEIGHT}">
      <joint name="agent_{i}_x" type="slide" axis="1 0 0" limited="false" damping="{damping}"/>
      <joint name="agent_{i}_y" type="slide" axis="0 1 0" limited="false" damping="{damping}"/>
      <geom type="cylinder" size="{_CYLINDER_RADIUS} {_CYLINDER_HALF_HEIGHT}" rgba="{r} {g} {b} 0"/>{_humanoid_geoms(r, g, b)}
    </body>"""
            motors_xml += f"""
    <motor name="agent_{i}_motor_x" joint="agent_{i}_x" gear="{max_force}" ctrlrange="-1 1"/>
    <motor name="agent_{i}_motor_y" joint="agent_{i}_y" gear="{max_force}" ctrlrange="-1 1"/>"""

        # Visible "stage": a dark danger ground with a lighter circular safe zone
        # (the arena). The safe-zone disk is named so the renderer can shrink it
        # each frame to visualise the closing storm.
        stage_xml = f"""
    <geom name="ground" type="plane" size="12 12 0.1" rgba="0.10 0.10 0.13 1"/>
    <geom name="safe_zone" type="cylinder" size="{arena_radius:.6f} 0.02" pos="0 0 0.01" rgba="0.30 0.52 0.68 1" contype="0" conaffinity="0"/>
    <geom name="safe_ring" type="cylinder" size="{arena_radius * 1.03:.6f} 0.015" pos="0 0 0.008" rgba="0.85 0.9 0.95 1" contype="0" conaffinity="0"/>"""

        return f"""<mujoco>
  <option timestep="0.01"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/>
  </visual>
  <worldbody>
    <light pos="0 -3 6" dir="0 0.4 -1" directional="true"/>
    <camera name="main" pos="0 -{arena_radius * 2.6:.4f} {arena_radius * 1.9:.4f}" xyaxes="1 0 0 0 0.6 0.8"/>{stage_xml}{bodies_xml}
  </worldbody>
  <actuator>{motors_xml}
  </actuator>
</mujoco>"""


# Backward-compatible module-level function for main.py
def build() -> str:
    return XMLBuilder.build(num_agents=1, arena_radius=3.0, max_force=10.0)
