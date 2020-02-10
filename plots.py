import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm, animation
from scipy.optimize import minimize
import utils
import seaborn as sns
import imageio
import time

def plot_3d(fn, x1_low, x1_high, x2_low, x2_high, stepsize=0.1):
    # Create 2d raster
    x1_steps = np.arange(x1_low, x1_high, stepsize)
    x2_steps = np.arange(x2_low, x2_high, stepsize)
    x1, x2 = np.meshgrid(x1_steps, x2_steps)
    
    # Plot
    y = fn(x1, x2)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x1, x2, y, cmap=cm.plasma, linewidth=0, antialiased=False)
    plt.show()

def visualize_heatmap(fn, history, extent, fname="particles.gif", output = "show"):
    fig = plt.figure()
    ax = plt.axes()
    
    # Create heatmap
    X = np.arange(extent[0], extent[1], 0.1)
    Y = np.arange(extent[2], extent[3], 0.1)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z = fn([X_grid, Y_grid])
    patch = plt.imshow(Z, extent=extent, cmap=cm.jet)
    fig.colorbar(patch, ax=ax)

    minimum = minimize(fn, [0, 0])
    ax.plot(minimum.x[0], minimum.x[1], "g*")

    average_x = np.mean([p["pos"][0] for p in history[0]])
    average_y = np.mean([p["pos"][1] for p in history[0]])
    ax.plot(average_x, average_y, "r*")
    
    # Create initial scatterplot
    x_points = [p["pos"][0] for p in history[0]]
    y_points = [p["pos"][1] for p in history[0]]
    sc = ax.scatter(x=x_points, y=y_points, color="black")
    
    # Create initial lineplots
    num_particles = len(history[0])
    lines = []
    for i in range(num_particles):
        lines.append(ax.plot(0, 0, color="blue")[0])
    
    # Function for animating scatterplot
    def animate(i):
        state = history[i]
        # update particles
        x_points = [p["pos"][0] for p in state]
        y_points = [p["pos"][1] for p in state]
        sc.set_offsets(np.c_[x_points,y_points])

        average_x = np.mean(x_points)
        average_y = np.mean(y_points)
        
        # update motion lines
        num_frames = min(20, i)
        x_steps = np.empty((num_particles, num_frames))
        y_steps = np.empty((num_particles, num_frames))

        for frame, all_particles in enumerate(history[i-num_frames:i]):
            for p_index, particle in enumerate(all_particles):
                x_steps[p_index, frame] = particle["pos"][0]
                y_steps[p_index, frame] = particle["pos"][1]
            
        for i, line in enumerate(lines):
            line.set_data(x_steps[i], y_steps[i])
    
    if(output == "step"):
        # Step through the frames
        global index
        index = 0       
        
        def on_keyboard(event):
            global index
            if event.key == 'right':
                if(index < len(history)):
                    index += 1
            elif event.key == 'left':
                if(index != 0):
                    index -= 1
                
            animate(index)

            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.gcf().canvas.mpl_connect('key_press_event', on_keyboard)
        plt.show()

    else:
        anim = animation.FuncAnimation(fig, animate, len(history), interval=20, blit=False)
        
        if(output == "show"):
            plt.show()
        elif(output == "save"):
            anim.save(fname, writer='imagemagick', fps=60)


def visualize_3D(fn, history):
    # TODO: this whole thing about the plot is still not quite right
    # (0 point is different)
    buffer = []
    for state in history:
        plt.close("all")
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        X = np.arange(-2, 2, 0.1)
        Y = np.arange(-2, 2, 0.1)
        X, Y = np.meshgrid(X, Y)
        a = 0
        b = 1000
        Z = utils.rastrigin(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0,
        antialiased=False)

        # visualize particles
        x_points = [i["pos"][0] for i in state]
        y_points = [i["pos"][1] for i in state]
        z_points = [i["fit"] for i in state]
        ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv')

        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        buffer.append(image)

    imageio.mimsave("particles.gif", buffer, )


if __name__ == "__main__":
    plot_3d(utils.rosenbrock, -2, 2, -2, 2)
    plot_3d(utils.rastrigin, -2, 2, -2, 2)
