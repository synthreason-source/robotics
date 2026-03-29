import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
import os
import io
from PIL import Image

# Ensure output directory exists
os.makedirs('output', exist_ok=True)
print("Starting Smooth 3D Multi-Plate Acoustic Simulation...")

# ---------------------------------------------------------
# 1. PARAMETERS
# ---------------------------------------------------------
L = 3            
N = 30           
frames = 800      
C2 = 0.15        
damping = 0.99   

start_pos = (0, 5, 5)     
target_pos = (2, 25, 25)  

u = np.zeros((L, N, N))
u_prev = np.zeros((L, N, N))
arrival_time = np.full((L, N, N), np.inf)
balls_mask = np.zeros((L, N, N), dtype=bool)

# --- COMPLEX OBSTACLES ---
balls_mask[0, 14:16, :] = True
balls_mask[0, 14:16, 6:10] = False  
balls_mask[0, 14:16, 20:24] = False 
balls_mask[0, 0:8, 22:30] = True    

balls_mask[1, 8:22, 14:16] = True
balls_mask[1, 14:16, 8:22] = True
balls_mask[1, 14:16, 14:16] = False 

balls_mask[2, 20:22, 15:30] = True
balls_mask[2, 15:30, 20:22] = True
balls_mask[2, 15:17, 22:28] = True
# -----------------------------

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
# 3. PATHFINDING
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

# ---------------------------------------------------------
# 4. 3D MULTI-PLATE RENDERING
# ---------------------------------------------------------
print("Rendering...")
X, Y = np.meshgrid(np.arange(N), np.arange(N))

# Fix colormap warning using modern Matplotlib API
colormap = mpl.colormaps['twilight_shifted']

norm = Normalize(vmin=-1.0, vmax=1.0)
Z_SPACING = 15

pil_frames = []

RENDER_FRAMES = 150
render_steps = np.linspace(0, frames-1, RENDER_FRAMES).astype(int)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('#050505')

for i in render_steps:
    ax.clear()
    ax.set_facecolor('#050505')
    
    wave_state = wave_frames[i]
    
    for z in range(L):
        Z_grid = np.full((N, N), z * Z_SPACING)
        colors = colormap(norm(wave_state[z]))
        colors[:, :, 3] = 0.5  
        colors[balls_mask[z]] = [0.1, 0.1, 0.1, 0.9]
        
        ax.plot_surface(X, Y, Z_grid, facecolors=colors, rstride=2, cstride=2, shade=False)
        ax.text(0, N, z * Z_SPACING + 2, f"Layer {z}", color='white', fontsize=6)

    if mic_hit_frame != -1 and i >= mic_hit_frame:
        progress = (i - mic_hit_frame) / max(1, (frames - 1) - mic_hit_frame)
        idx = int(progress * len(path)) + 1 
        idx = min(idx, len(path)) 
        
        if idx > 1:
            p_x = [p[2] for p in path[:idx]]
            p_y = [p[1] for p in path[:idx]]
            p_z = [p[0] * Z_SPACING + 1.0 for p in path[:idx]]
            ax.plot(p_x, p_y, p_z, color='#00ffcc', linewidth=3, zorder=10)
            
    ax.scatter([start_pos[2]], [start_pos[1]], [start_pos[0] * Z_SPACING + 2], color='#00ffcc', s=80, marker='*')
    ax.scatter([target_pos[2]], [target_pos[1]], [target_pos[0] * Z_SPACING + 2], color='#ff00cc', s=80, marker='X')
    
    ax.set_zlim(-5, (L-1)*Z_SPACING + 5)
    
    ax.view_init(elev=15, azim=-60 + (i * 0.2)) 
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#050505', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # FIX: Use .copy() to completely detach the image from the memory buffer
    img = Image.open(buf).copy()
    pil_frames.append(img)
    buf.close()
    
    print(f"Rendered frame {i}/{frames}", end="\r")

gif_path = 'output/smooth_acoustic_multiplate.gif'

pause_duration_seconds = 2.0  
frame_duration_ms = 30
extra_frames = int((pause_duration_seconds * 1000) / frame_duration_ms)

if pil_frames:
    pil_frames.extend([pil_frames[-1]] * extra_frames)

# Now it can safely save because all images are fully loaded and copied in memory
pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=frame_duration_ms, loop=0)

print(f"\nDone! Saved smooth animation to {gif_path}")
