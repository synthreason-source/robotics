import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import os
import io
from PIL import Image

os.makedirs('output/sims', exist_ok=True)
print("Starting Optimized 3D Multi-Plate Acoustic Simulation...")

# ---------------------------------------------------------
# 1. PARAMETERS
# ---------------------------------------------------------
L = 3            
N = 30           # Lower resolution to prevent timeout
frames = 250      # Fewer frames to prevent timeout
C2 = 0.15        
damping = 0.99   

start_pos = (0, 5, 5)     
target_pos = (2, 25, 25)  

u = np.zeros((L, N, N))
u_prev = np.zeros((L, N, N))
arrival_time = np.full((L, N, N), np.inf)
balls_mask = np.zeros((L, N, N), dtype=bool)

# Plate 0: Huge wall blocking X-axis
balls_mask[0, :, 12:18] = True  
# Plate 1: Huge wall blocking Y-axis
balls_mask[1, 12:18, :] = True  
# Plate 2: Blocking corner
balls_mask[2, 15:25, 10:20] = True 

# ---------------------------------------------------------
# 2. 3D PHYSICS ENGINE 
# ---------------------------------------------------------
wave_frames = []
mic_hit_frame = -1

for step in range(frames):
    u_pad = np.pad(u, 1, mode='constant')
    laplacian = (
        u_pad[2:, 1:-1, 1:-1] + u_pad[:-2, 1:-1, 1:-1] + 
        u_pad[1:-1, 2:, 1:-1] + u_pad[1:-1, :-2, 1:-1] + 
        u_pad[1:-1, 1:-1, 2:] + u_pad[1:-1, 1:-1, :-2]   
        - 6 * u
    )
    
    u_next = 2*u - u_prev + C2 * laplacian
    u_next *= damping
    
    if step < 15: 
        u_next[start_pos] += np.sin(step * np.pi / 4.0) * 10.0
        
    u_next[balls_mask] = 0.0
    
    mask = (np.abs(u_next) > 0.1) & (arrival_time == np.inf)
    arrival_time[mask] = step
    
    if arrival_time[target_pos] != np.inf and mic_hit_frame == -1:
        mic_hit_frame = step
        print(f"Target Hit at frame {step}!")
        
    u_prev = u.copy()
    u = u_next.copy()
    wave_frames.append(u.copy())

# ---------------------------------------------------------
# 3. 3D PATHFINDING ALGORITHM
# ---------------------------------------------------------
path = [target_pos]
curr = target_pos
visited = set([target_pos])

if mic_hit_frame != -1:
    while curr != start_pos:
        neighbors = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz==0 and dy==0 and dx==0: continue
                    nz, ny, nx = curr[0]+dz, curr[1]+dy, curr[2]+dx
                    if 0<=nz<L and 0<=ny<N and 0<=nx<N:
                        neighbors.append((nz, ny, nx))
                        
        valid = [n for n in neighbors if n not in visited and arrival_time[n] != np.inf]
        if not valid: break
        
        curr = min(valid, key=lambda x: arrival_time[x])
        path.append(curr)
        visited.add(curr)
        if len(path) > L*N*N: break

path = path[::-1] 
print(f"Path calculated.")

# ---------------------------------------------------------
# 4. 3D MULTI-PLATE RENDERING
# ---------------------------------------------------------
print("Rendering...")
X, Y = np.meshgrid(np.arange(N), np.arange(N))
colormap = cm.get_cmap('twilight_shifted')
norm = Normalize(vmin=-1.0, vmax=1.0)
Z_SPACING = 15

pil_frames = []
for i in range(frames):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050505')
    fig.patch.set_facecolor('#050505')
    
    wave_state = wave_frames[i]
    
    for z in range(L):
        Z_grid = np.full((N, N), z * Z_SPACING)
        colors = colormap(norm(wave_state[z]))
        colors[:, :, 3] = 0.5  # Transparency
        colors[balls_mask[z]] = [0.1, 0.1, 0.1, 0.9]
        ax.plot_surface(X, Y, Z_grid, facecolors=colors, rstride=1, cstride=1, shade=False)
        ax.text(0, N, z * Z_SPACING + 2, f"Layer {z}", color='white', fontsize=6)

    if mic_hit_frame != -1 and i >= mic_hit_frame:
        idx = int((i - mic_hit_frame) * (len(path) / max(1, frames - mic_hit_frame)))
        if idx > 0:
            p_x = [p[2] for p in path[:idx]]
            p_y = [p[1] for p in path[:idx]]
            p_z = [p[0] * Z_SPACING + 1.0 for p in path[:idx]]
            ax.plot(p_x, p_y, p_z, color='#00ffcc', linewidth=3, zorder=10)
            
    ax.scatter([start_pos[2]], [start_pos[1]], [start_pos[0] * Z_SPACING + 2], color='#00ffcc', s=80, marker='*')
    ax.scatter([target_pos[2]], [target_pos[1]], [target_pos[0] * Z_SPACING + 2], color='#ff00cc', s=80, marker='X')
    
    ax.set_zlim(-5, (L-1)*Z_SPACING + 5)
    ax.view_init(elev=15, azim=-60 + (i * 0.8)) 
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#050505', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    pil_frames.append(Image.open(buf))
    plt.close(fig)

gif_path = 'acoustic_multiplate.gif'
pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=50, loop=0)
print(f"Done! Saved to {gif_path}")