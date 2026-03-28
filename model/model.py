
import mujoco
xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""


def load_model(model_string: str = "", model_path: str = "") -> mujoco.MjModel:
    if model_string:
        model = mujoco.MjModel.from_xml_string(model_string)
    elif model_path:
        model = mujoco.MjModel.from_xml_path(model_path)
    return model


def load_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    return data