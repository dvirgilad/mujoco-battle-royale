
import mujoco
import mediapy as media


def load_model(model_string: str = "", model_path: str = "") -> mujoco.MjModel:
    if model_string:
        model = mujoco.MjModel.from_xml_string(model_string)
    elif model_path:
        model = mujoco.MjModel.from_xml_path(model_path)
    else:
        raise ValueError("Either model_string or model_path must be provided.")
    return model


def load_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    return data


def render(model: mujoco.MjModel, data: mujoco.MjData, output_path: str = "output.png") -> None:
    with mujoco.Renderer(model) as renderer:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        media.write_image(output_path, renderer.render())