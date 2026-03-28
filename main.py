import os
import mediapy as media
from model.model import load_model, load_data

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
if __name__ == "__main__":
    from model.model import xml
    import mujoco

    model = load_model(model_path="model/model.xml")
    data = load_data(model)
    with mujoco.Renderer(model) as renderer:
        media.write_image("output.png", renderer.render())