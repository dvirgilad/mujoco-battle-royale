import os
from model import load_model, load_data, render

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
if __name__ == "__main__":


    model = load_model(model_path="model/model.xml")
    data = load_data(model)
    render(model, data, "output.png")