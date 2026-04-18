

def build() -> str:
    '''Builds the XML string for the Mujoco model.'''
    return """
        <mujoco>
          <worldbody>
             <body name="colored_circle" pos="0 0 0.5">
                <geom type="sphere" size="0.105" rgba="1 0 0 1" /> <!-- Red Outline -->
            
                <geom type="sphere" size="0.1" rgba="0 1 0 1" /> <!-- Green Fill -->
             </body>
            </worldbody>
        </mujoco>
    """

