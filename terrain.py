import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from matplotlib.colors import LightSource
import matplotlib.cm as cm
import threading
import os

size = 250

# Color palette from the image
PALETTE = {
    "bg": "#fdf1e6",         # top (background)
    "panel1": "#c3cef6",     # second
    "panel2": "#a3b1dc",     # third
    "panel3": "#8f9fd1",     # fourth
    "panel4": "#868dc1",     # bottom
    "button_fg": "#232946"
}

def generate_terrain(terrain_type):
    if terrain_type == "Mountains":
        octaves, scale = 6, 1
        def postprocess(t): return 1 - t
    elif terrain_type == "Hills":
        octaves, scale = 4, 0.88
        def postprocess(t): return t ** 1.5
    elif terrain_type == "Plateau":
        octaves, scale = 5, 0.19
        def postprocess(t):
            t = (t - t.min()) / (t.max() - t.min())
            return t ** 10 # 0.1% bumps, centered at 0.5
    elif terrain_type == "Island":
        octaves, scale = 5, 1.5
        def postprocess(t):
            cx, cy = size // 2, size // 2
            yy, xx = np.meshgrid(np.arange(size), np.arange(size))
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size / 2)
            t *= np.clip(1 - dist, 0, 1)
            return t
    else:
        octaves, scale = 6, 1
        def postprocess(t): return t

    noise = PerlinNoise(octaves=octaves, seed=42)
    terrain = np.zeros((size, size))

    # Use threading for terrain generation
    num_threads = os.cpu_count() or 4
    threads = []
    rows_per_thread = size // num_threads

    def worker(start, end):
        for i in range(start, end):
            for j in range(size):
                terrain[i][j] = noise([i / size * scale, j / size * scale])

    for t_idx in range(num_threads):
        start = t_idx * rows_per_thread
        end = (t_idx + 1) * rows_per_thread if t_idx != num_threads - 1 else size
        thread = threading.Thread(target=worker, args=(start, end))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    terrain = postprocess(terrain)
    return terrain, num_threads

def generate_biome_map(terrain):
    biome_map = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            h = terrain[i][j]
            if h < 0.3:
                biome_map[i][j] = [0.0, 0.4, 0.8]       # Deep Water
            elif h < 0.4:
                biome_map[i][j] = [0.9, 0.8, 0.5]       # Sand
            elif h < 0.6:
                biome_map[i][j] = [0.1, 0.8, 0.1]       # Grass
            elif h < 0.75:
                biome_map[i][j] = [0.1, 0.5, 0.1]       # Forest
            elif h < 0.9:
                biome_map[i][j] = [0.5, 0.5, 0.5]       # Rock
            else:
                biome_map[i][j] = [1.0, 1.0, 1.0]       # Snow
    return biome_map

def show_3d_terrain(terrain, terrain_type):
    x = np.linspace(0, 0.5, size)
    y = np.linspace(0, 1, size)
    x, y = np.meshgrid(x, y)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(terrain, cmap=cm.get_cmap('gist_earth'), vert_exag=1, blend_mode='soft')
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, terrain, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(f"3D Terrain: {terrain_type}", fontsize=16)
    plt.tight_layout()
    plt.show()

def show_biome_map(terrain):
    biome_map = generate_biome_map(terrain)
    plt.figure(figsize=(6, 6))
    plt.imshow(biome_map, origin='lower')
    plt.title("Biome Map")
    plt.axis('off')
    plt.show()

def show_height_map(terrain):
    plt.figure(figsize=(6, 6))
    plt.imshow(terrain, cmap='terrain', origin='lower')
    plt.title("Height Map")
    plt.axis('off')
    plt.colorbar(shrink=0.7)
    plt.show()

def show_temperature_map(terrain):
    temperature_map = 1.0 - terrain
    plt.figure(figsize=(6, 6))
    plt.imshow(temperature_map, cmap='coolwarm', origin='lower')
    plt.title("Temperature Map")
    plt.axis('off')
    plt.colorbar(shrink=0.7)
    plt.show()

def show_moisture_map(terrain):
    moisture_map = (1 - terrain) ** 2
    plt.figure(figsize=(6, 6))
    plt.imshow(moisture_map, cmap='Blues', origin='lower')
    plt.title("Moisture Map")
    plt.axis('off')
    plt.colorbar(shrink=0.7)
    plt.show()

def show_topographic_map(terrain):
    plt.figure(figsize=(6, 6))
    plt.contourf(terrain, cmap='Greens')
    plt.title("Topographic Map")
    plt.axis('off')
    plt.show()

def on_generate(terrain_type):
    global current_terrain, threads_used
    current_terrain, threads_used = generate_terrain(terrain_type)
    show_3d_terrain(current_terrain, terrain_type)

def on_biome():
    if current_terrain is not None:
        show_biome_map(current_terrain)

def on_height():
    if current_terrain is not None:
        show_height_map(current_terrain)

def on_temp():
    if current_terrain is not None:
        show_temperature_map(current_terrain)

def on_moisture():
    if current_terrain is not None:
        show_moisture_map(current_terrain)

def on_topo():
    if current_terrain is not None:
        show_topographic_map(current_terrain)

def on_exit():
    root.destroy()

def show_user_guide():
    guide = (
        "User Guide:\n\n"
        "• Click a terrain button in the left column to generate a 3D terrain.\n"
        "• Use the center map buttons to view:\n"
        "   - Biome Map: Shows terrain biomes (water, sand, grass, etc).\n"
        "   - Height Map: Shows elevation as color.\n"
        "   - Temperature Map: Simulated temperature based on height.\n"
        "   - Moisture Map: Simulated moisture based on height.\n"
        "   - Topographic Map: Contour lines of elevation.\n"
        "• The right column has extra options, including thread info and exit.\n"
        "• Click 'Threads/Complexity' to see how many threads are used and the code's complexity."
    )
    messagebox.showinfo("User Guide", guide)
def show_threads_info():
    info = (
        f"Threads used: {threads_used}\n"
        "Parallel code complexity: O(N^2) for terrain generation (N = grid size), "
        "but work is divided among threads for faster execution.\n\n"
        "If using serial code (no threads):\n"
        "• Time complexity remains O(N^2),\n"
        "• But all work is done by a single core, so execution is much slower.\n"
        "• Parallel code reduces wall-clock time by utilizing multiple CPU cores."
    )
    messagebox.showinfo("Threads & Complexity", info)

# --- GUI Setup ---
root = tk.Tk()
root.title("🌍 Terrain Generator")
root.geometry("900x540")
root.configure(bg=PALETTE["bg"])

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton",
                font=("Segoe UI", 12, "bold"),
                padding=8,
                background=PALETTE["panel2"],
                foreground=PALETTE["button_fg"],
                borderwidth=0)
style.map("TButton",
          background=[("active", PALETTE["panel3"])],
          foreground=[("active", PALETTE["button_fg"])])

# Main layout frames
main_frame = tk.Frame(root, bg=PALETTE["bg"])
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Left column: Terrain buttons
left_frame = tk.Frame(main_frame, bg=PALETTE["panel2"], bd=2, relief="ridge")
left_frame.pack(side="left", fill="y", padx=(0, 20), pady=10)

terrains_label = tk.Label(left_frame, text="Terrains", font=("Segoe UI", 16, "bold"),
                         bg=PALETTE["panel2"], fg=PALETTE["button_fg"])
terrains_label.pack(pady=(15, 15))

btn_width = 18

mountain_btn = ttk.Button(left_frame, text="Generate Mountain", width=btn_width, command=lambda: on_generate("Mountains"))
mountain_btn.pack(pady=8, ipadx=2, ipady=2)

hill_btn = ttk.Button(left_frame, text="Generate Hill", width=btn_width, command=lambda: on_generate("Hills"))
hill_btn.pack(pady=8, ipadx=2, ipady=2)

plain_btn = ttk.Button(left_frame, text="Generate Plateau/Plain", width=btn_width, command=lambda: on_generate("Plateau"))
plain_btn.pack(pady=8, ipadx=2, ipady=2)

island_btn = ttk.Button(left_frame, text="Generate Island", width=btn_width, command=lambda: on_generate("Island"))
island_btn.pack(pady=8, ipadx=2, ipady=2)

# Center column: Map buttons
center_frame = tk.Frame(main_frame, bg=PALETTE["bg"])
center_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

maps_heading = tk.Label(center_frame, text="Generate Maps", font=("Segoe UI", 16, "bold"),
                        bg=PALETTE["bg"], fg=PALETTE["button_fg"])
maps_heading.pack(pady=(15, 15))

maps_frame = tk.Frame(center_frame, bg=PALETTE["bg"])
maps_frame.pack(pady=10)

map_btn_width = 16

biome_btn = ttk.Button(maps_frame, text="Biome Map", width=map_btn_width, command=on_biome)
biome_btn.grid(row=0, column=0, padx=8, pady=8, ipadx=2, ipady=2)

height_btn = ttk.Button(maps_frame, text="Height Map", width=map_btn_width, command=on_height)
height_btn.grid(row=0, column=1, padx=8, pady=8, ipadx=2, ipady=2)

temp_btn = ttk.Button(maps_frame, text="Temperature Map", width=map_btn_width, command=on_temp)
temp_btn.grid(row=1, column=0, padx=8, pady=8, ipadx=2, ipady=2)

moisture_btn = ttk.Button(maps_frame, text="Moisture Map", width=map_btn_width, command=on_moisture)
moisture_btn.grid(row=1, column=1, padx=8, pady=8, ipadx=2, ipady=2)

topo_btn = ttk.Button(maps_frame, text="Topographic Map", width=map_btn_width, command=on_topo)
topo_btn.grid(row=2, column=0, columnspan=2, padx=8, pady=8, ipadx=2, ipady=2)

# Right column: Extra buttons
right_frame = tk.Frame(main_frame, bg=PALETTE["panel3"], bd=2, relief="ridge")
right_frame.pack(side="left", fill="y", padx=(20, 0), pady=10)

extra_label = tk.Label(right_frame, text="Extra", font=("Segoe UI", 16, "bold"),
                      bg=PALETTE["panel3"], fg=PALETTE["button_fg"])
extra_label.pack(pady=(15, 15))

guide_btn = ttk.Button(right_frame, text="❓ User Guide", width=btn_width, command=show_user_guide)
guide_btn.pack(pady=12, ipadx=2, ipady=2)

threads_btn = ttk.Button(right_frame, text="Threads/Complexity", width=btn_width, command=show_threads_info)
threads_btn.pack(pady=12, ipadx=2, ipady=2)

exit_btn = ttk.Button(right_frame, text="❌ Exit", width=btn_width, command=on_exit)
exit_btn.pack(pady=12, ipadx=2, ipady=2)

current_terrain = None
threads_used = os.cpu_count() or 4

root.mainloop()