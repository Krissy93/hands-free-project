import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

def main():
    # Inizializza GLFW
    if not glfw.init():
        print("Impossibile inizializzare GLFW")
        return
    
    # Crea una finestra
    window = glfw.create_window(800, 600, "Test OpenGL", None, None)
    if not window:
        print("Impossibile creare la finestra")
        glfw.terminate()
        return
    
    glfw.make_context_current(window)

    # Definisce i vertici e i colori del triangolo
    vertices = [
        0.0,  0.5, 0.0,  # Vertice 1
       -0.5, -0.5, 0.0,  # Vertice 2
        0.5, -0.5, 0.0   # Vertice 3
    ]
    
    colors = [
        1.0, 0.0, 0.0,  # Colore rosso per il vertice 1
        0.0, 1.0, 0.0,  # Colore verde per il vertice 2
        0.0, 0.0, 1.0   # Colore blu per il vertice 3
    ]
    
    # Configura i buffer
    vertex_buffer = glGenBuffers(1)
    color_buffer = glGenBuffers(1)
    
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, 4 * len(vertices), (GLfloat * len(vertices))(*vertices), GL_STATIC_DRAW)
    
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer)
    glBufferData(GL_ARRAY_BUFFER, 4 * len(colors), (GLfloat * len(colors))(*colors), GL_STATIC_DRAW)
    
    # Imposta il layout dei buffer
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glVertexPointer(3, GL_FLOAT, 0, None)
    
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer)
    glColorPointer(3, GL_FLOAT, 0, None)

    # Loop di rendering
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Disegna il triangolo
        glDrawArrays(GL_TRIANGLES, 0, 3)
        
        # Swap dei buffer
        glfw.swap_buffers(window)
        
        # Polling degli eventi
        glfw.poll_events()

    # Pulizia
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDeleteBuffers(1, [vertex_buffer])
    glDeleteBuffers(1, [color_buffer])
    glfw.terminate()

if __name__ == "__main__":
    main()
