import time
import mujoco
from mujoco import viewer
import mujoco_viewer
from src.battle_royale.infrastructure.physics.xml_builder import build
def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build())

def load_data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)

def render(model: mujoco.MjModel, data: mujoco.MjData, output_path: str) -> None:
    '''Renders the Mujoco model and saves the output image.
    Args:
        model: The Mujoco model to render.
        data: The Mujoco data associated with the model.
        output_path: The file path to save the rendered image.
    '''
    with mujoco.Renderer(model) as renderer:
        renderer.update_scene(data)
        renderer.render()
        viewer = mujoco_viewer.MujocoViewer(model, data)
        try:
            while viewer.is_alive:
                mujoco.mj_step(model, data)
                viewer.render()
        # Optional: add a small pause to control simulation speed
            time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()
