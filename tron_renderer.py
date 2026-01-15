import pygame
import moderngl as mgl
import numpy as np

class TronRenderer:
    def __init__(self, grid_w, grid_h):
        self.grid_w = grid_w
        self.grid_h = grid_h
        
        # 1. Init Pygame
        # We check if pygame is already initialized to avoid errors
        if not pygame.get_init():
            pygame.init()
            
        pygame.display.set_mode((800, 800), flags=pygame.OPENGL | pygame.DOUBLEBUF)
        
        # 2. Init ModernGL
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.BLEND)
        
        # 3. Compile Shaders (Same as before)
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert;
                in vec2 in_uv;
                out vec2 uv;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D GameGrid;
                uniform float time;
                in vec2 uv;
                out vec4 color;
                
                void main() {
                    float val = texture(GameGrid, uv).r * 255.0;
                    
                    // Base Color (Dark Floor)
                    vec3 pixel = vec3(0.05, 0.05, 0.08); 
                    
                    // Colors
                    if (val > 0.9 && val < 1.1)      pixel = vec3(0.5); // Wall
                    else if (val > 1.9 && val < 2.1) pixel = vec3(1.0, 1.0, 0.0); // Boost
                    else if (val > 2.9 && val < 3.1) pixel = vec3(0.0, 1.0, 1.0); // P1 Head
                    else if (val > 3.9 && val < 4.1) pixel = vec3(1.0, 0.0, 0.2); // P2 Head
                    else if (val > 4.9 && val < 5.1) pixel = vec3(0.0, 0.5, 0.5); // P1 Trail
                    else if (val > 5.9 && val < 6.1) pixel = vec3(0.5, 0.0, 0.1); // P2 Trail
                    
                    // Grid Lines
                    vec2 gridPos = uv * vec2(100.0, 100.0);
                    vec2 gridSt = fract(gridPos);
                    float edge = step(0.95, gridSt.x) + step(0.95, gridSt.y);
                    pixel = mix(pixel, vec3(0.2), clamp(edge, 0.0, 1.0));

                    // Glow
                    if (val > 1.9 && val < 4.1) {
                        pixel *= (1.2 + 0.3 * sin(time * 8.0));
                    }

                    color = vec4(pixel, 1.0);
                }
            """
        )
        
        # 4. Geometry
        vertices = np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, -1,1,0,1, 1,-1,1,0, 1,1,1,1], dtype='f4')
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f 2f', 'in_vert', 'in_uv')])
        self.texture = self.ctx.texture((grid_w, grid_h), 1, dtype='f1')
        self.texture.filter = (mgl.NEAREST, mgl.NEAREST)

    def render_frame(self, grid_data):
        # Write Data
        self.texture.write(grid_data.astype('u1').tobytes())
        self.texture.use(0)
        self.prog['time'].value = pygame.time.get_ticks() / 1000.0
        
        # Draw
        self.ctx.clear()
        self.vao.render()
        pygame.display.flip()
        
        # CRITICAL FIX: Return True so TronEnv knows the window is still open
        return True
        
    def close(self):
        pygame.quit()