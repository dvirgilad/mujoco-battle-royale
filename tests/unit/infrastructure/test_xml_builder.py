import math
import xml.etree.ElementTree as ET

from battle_royale.infrastructure.physics.xml_builder import XMLBuilder


def parse(xml_str: str) -> ET.Element:
    return ET.fromstring(xml_str.strip())


def test_xml_builder_produces_valid_xml():
    xml_str = XMLBuilder.build(num_agents=2, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    assert root.tag == "mujoco"


def test_xml_builder_has_floor_plane():
    xml_str = XMLBuilder.build(num_agents=2, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    worldbody = root.find("worldbody")
    assert worldbody is not None
    plane = worldbody.find(".//geom[@type='plane']")
    assert plane is not None


def test_xml_builder_creates_correct_number_of_agents():
    for n in [2, 4, 6, 8]:
        xml_str = XMLBuilder.build(num_agents=n, arena_radius=3.0, max_force=10.0)
        root = parse(xml_str)
        bodies = root.findall(".//body[@name]")
        agent_bodies = [b for b in bodies if b.get("name", "").startswith("agent_")]
        assert len(agent_bodies) == n, f"Expected {n} agents, got {len(agent_bodies)}"


def test_xml_builder_agent_cylinders():
    xml_str = XMLBuilder.build(num_agents=4, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    geoms = root.findall(".//geom[@type='cylinder']")
    assert len(geoms) == 4


def test_xml_builder_agents_have_slide_joints():
    xml_str = XMLBuilder.build(num_agents=3, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    # Each agent body should have 2 slide joints (x and y)
    for i in range(3):
        body = root.find(f".//body[@name='agent_{i}']")
        assert body is not None
        joints = body.findall("joint")
        assert len(joints) == 2
        axes = {j.get("axis") for j in joints}
        assert "1 0 0" in axes
        assert "0 1 0" in axes


def test_xml_builder_motors_per_agent():
    xml_str = XMLBuilder.build(num_agents=4, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    actuators = root.find("actuator")
    assert actuators is not None
    motors = actuators.findall("motor")
    # 2 motors per agent
    assert len(motors) == 8


def test_xml_builder_spawn_positions_on_ring():
    n = 4
    arena_radius = 3.0
    xml_str = XMLBuilder.build(num_agents=n, arena_radius=arena_radius, max_force=10.0)
    root = parse(xml_str)
    spawn_r = 0.6 * arena_radius  # 1.8
    for i in range(n):
        body = root.find(f".//body[@name='agent_{i}']")
        assert body is not None
        pos_str = body.get("pos")
        assert pos_str is not None
        parts = pos_str.split()
        x, y = float(parts[0]), float(parts[1])
        actual_r = math.sqrt(x**2 + y**2)
        assert abs(actual_r - spawn_r) < 1e-4, f"Agent {i} spawn radius {actual_r} != {spawn_r}"


def test_xml_builder_agent_colors_cycle():
    xml_str = XMLBuilder.build(num_agents=4, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    expected_colors = [
        (1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0),
    ]
    for i, (r, g, b) in enumerate(expected_colors):
        body = root.find(f".//body[@name='agent_{i}']")
        geom = body.find("geom")
        rgba = geom.get("rgba")
        parts = rgba.split()
        assert abs(float(parts[0]) - r) < 1e-3
        assert abs(float(parts[1]) - g) < 1e-3
        assert abs(float(parts[2]) - b) < 1e-3


def test_xml_builder_qpos_layout_2i_convention():
    # Verifies joint naming convention so qpos[2*i]=x, qpos[2*i+1]=y
    xml_str = XMLBuilder.build(num_agents=3, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    for i in range(3):
        body = root.find(f".//body[@name='agent_{i}']")
        joints = body.findall("joint")
        names = {j.get("name") for j in joints}
        assert f"agent_{i}_x" in names
        assert f"agent_{i}_y" in names
