from src.battle_royale.infrastructure.physics import load_data, load_model, render


if __name__ == "__main__":
    import glfw
    import os

    os.environ["XDG_SESSION_TYPE"] = "x11" #
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    if not glfw.init():
        print("Failed to initialize GLFW")

        model = load_model()
        data = load_data(model)
        render(model, data, "output.png")
    
    glfw.terminate()
