import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from noise import pnoise3

# --- параметры ---
WIDTH, HEIGHT = 1920, 1080
FPS = 30
DURATION = 12  # секунд (делай больше для менее заметного лупа)

GRID_X = 250
GRID_Y = 140

SCALE = 0.05      # масштаб шума (чем меньше — тем плавнее волны)
SPEED = 0.02      # скорость анимации
AMPLITUDE = 2.5   # высота волн

# --- создаём сетку ---
x = np.linspace(0, GRID_X, GRID_X)
y = np.linspace(0, GRID_Y, GRID_Y)
X, Y = np.meshgrid(x, y)

# нормализуем координаты
Xn = X * SCALE
Yn = Y * SCALE

# --- фигура ---
fig, ax = plt.subplots(figsize=(WIDTH/100, HEIGHT/100), dpi=100)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_facecolor("black")
ax.axis("off")

ax.set_xlim(-5, GRID_X + 5)
ax.set_ylim(-5, GRID_Y + 5)

# цветовая карта для градиента фиолетового
scatter = ax.scatter([], [], s=4, c=[], cmap="twilight_shifted", alpha=0.95, vmin=-1.5, vmax=1.5)

# --- функция шума с несколькими слоями ---
def generate_frame(t):
    Z = np.zeros_like(Xn)

    for i in range(Xn.shape[0]):
        for j in range(Xn.shape[1]):
            # несколько слоёв шума для более интересного паттерна
            noise1 = pnoise3(
                Xn[i][j],
                Yn[i][j],
                t * 0.5,
                octaves=3,
                persistence=0.6,
                lacunarity=2.0
            )
            
            noise2 = pnoise3(
                Xn[i][j] * 2,
                Yn[i][j] * 2,
                t * 0.8,
                octaves=2,
                persistence=0.5,
                lacunarity=2.2
            )
            
            # комбинируем шумы
            Z[i][j] = noise1 * 0.6 + noise2 * 0.4

    return Z

# --- анимация ---
def update(frame):
    t = frame * SPEED

    Z = generate_frame(t)

    # создаём "волновую поверхность" с двумерным смещением
    X_offset = X + Z * AMPLITUDE * 10
    Y_offset = Y + Z * AMPLITUDE * 12

    # нормализуем Z для цветовой схемы
    Z_normalized = np.clip(Z, -1.5, 1.5)

    scatter.set_offsets(np.c_[X_offset.flatten(), Y_offset.flatten()])
    scatter.set_array(Z_normalized.flatten())

    return scatter,

frames = FPS * DURATION

ani = animation.FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=1000/FPS,
    blit=True,
    repeat=True
)

# --- сохраняем ---
ani.save("background.mp4", fps=FPS, dpi=100)

print("Готово: background.mp4")