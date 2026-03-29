import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# ---------------------------------------------------------
# 1. SIMULATION PARAMETERS
# ---------------------------------------------------------
N = 120          # Grid resolution (120x120 plate)
frames = 600     # Total frames for animation
c = 1.0          # Wave speed
dt = 0.5         # Time step
dx = 1.0         # Spatial step
C2 = (c * dt / dx)**2
damping = 0.995  # Acoustic damping

start_pos = (15, 15)      # Acoustic Actuator
target_pos = (105, 105)   # Target Microphone

# Matrices
u = np.zeros((N, N))
u_prev = np.zeros((N, N))
arrival_time = np.full((N, N), np.inf)

# ---------------------------------------------------------
# 2. HARDWARE ABSTRACTION: MAGNETIC BALLS ON SPRINGS
# ---------------------------------------------------------
# We define specific regions where electromagnets lock the 
# spring-mounted balls to the plate, solidifying the surface 
# and blocking wave propagation.
balls_mask = np.zeros((N, N), dtype=bool)

def add_ball_cluster(cx, cy, radius):
    for i in range(N):
        for j in range(N):
            if (i - cx)**2 + (j - cy)**2 <= radius**2:
                # Add individual "balls" within the radius to simulate granular locking
                if np.random.rand() > 0.1: 
                    balls_mask[i, j] = True

# Create a maze-like structure of solidified balls
add_ball_cluster(40, 30, 20)
add_ball_cluster(40, 80, 25)
add_ball_cluster(80, 40, 25)
add_ball_cluster(90, 85, 15)
add_ball_cluster(20, 100, 15)

# ---------------------------------------------------------
# 3. PHYSICS ENGINE (2D Wave Equation)
# ---------------------------------------------------------
wave_frames = []
mic_hit_frame = -1

print("Simulating 2D Acoustic Plate Physics...")
for step in range(frames):
    # Finite Difference Laplacian
    laplacian = (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
    )
    
    # Wave Equation: u_tt = c^2 * Laplacian
    u_next = 2 * u - u_prev + C2 * laplacian
    
    # Damping
    u_next *= damping
    
    # Hard boundaries (Plate Edges)
    u_next[0, :] = 0; u_next[-1, :] = 0
    u_next[:, 0] = 0; u_next[:, -1] = 0
    
    # Inject acoustic pulse at Actuator
    if step < 25:
        u_next[start_pos] += np.sin(step * np.pi / 8.0) * 8.0
        
    # Apply Solidified Balls (Dirichlet boundary condition u=0)
    u_next[balls_mask] = 0.0
    
    # Track First-Arrival Time (Time-of-Flight pathfinding)
    wave_front_mask = (u_next > 0.05) & (arrival_time == np.inf)
    arrival_time[wave_front_mask] = step
    
    # Check if microphone detected the wave
    if arrival_time[target_pos] != np.inf and mic_hit_frame == -1:
        mic_hit_frame = step
        print(f"-> Target Microphone detected wave at frame {step}!")
    
    # Step forward
    u_prev = u.copy()
    u = u_next.copy()
    wave_frames.append(u.copy())

# ---------------------------------------------------------
# 4. PATHFINDING ALGORITHM (Time-of-Flight Gradient Descent)
# ---------------------------------------------------------
# Backtrack from Microphone to Actuator using the arrival time map
print("Reconstructing optimal acoustic path...")
path = [target_pos]
curr = target_pos
visited = set([target_pos])

# Simple heuristic backtracking: step to the neighbor with the lowest arrival time
while abs(curr[0] - start_pos[0]) + abs(curr[1] - start_pos[1]) > 0:
    neighbors = [
        (curr[0]+1, curr[1]), (curr[0]-1, curr[1]), 
        (curr[0], curr[1]+1), (curr[0], curr[1]-1),
        (curr[0]+1, curr[1]+1), (curr[0]-1, curr[1]-1),
        (curr[0]+1, curr[1]-1), (curr[0]-1, curr[1]+1)
    ]
    
    valid_neighbors = []
    for n in neighbors:
        if 0 <= n[0] < N and 0 <= n[1] < N and n not in visited:
            if arrival_time[n] != np.inf:
                valid_neighbors.append(n)
                
    if not valid_neighbors:
        break # Path stalled (shouldn't happen if wave arrived)
        
    # Move to the neighboring grid cell that the wave reached FIRST
    curr = min(valid_neighbors, key=lambda x: arrival_time[x])
    path.append(curr)
    visited.add(curr)

path = path[::-1] # Reverse to go Start -> Target
path_x = [p[1] for p in path]
path_y = [p[0] for p in path]
print(f"Path successfully generated with {len(path)} nodes.")

# ---------------------------------------------------------
# 5. ANIMATION GENERATION
# ---------------------------------------------------------
print("Rendering Animation...")
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('#000000')

# Render wave
img = ax.imshow(wave_frames[0], cmap='twilight_shifted', vmin=-1.5, vmax=1.5, origin='lower')

# Render Magnetic Balls
balls_y, balls_x = np.where(balls_mask)
ax.scatter(balls_x, balls_y, color='#111111', edgecolor='#333333', s=10, label='Locked Magnetic Balls')

# Render Start and Target
ax.scatter([start_pos[1]], [start_pos[0]], color='#00ffcc', s=100, marker='*', label='Acoustic Actuator')
ax.scatter([target_pos[1]], [target_pos[0]], color='#ff00cc', s=100, marker='X', label='Target Microphone')

# Render Path line (initialized empty)
line, = ax.plot([], [], color='#00ffcc', lw=3, label='Acoustic Trajectory')

ax.set_title("Acoustic Time-of-Flight Pathfinding", color='white')
ax.legend(facecolor='#222222', labelcolor='white', loc='upper right')
ax.set_xticks([])
ax.set_yticks([])

def update(frame):
    img.set_data(wave_frames[frame])
    
    # If the wave has reached the microphone, dynamically draw the path line
    if mic_hit_frame != -1 and frame >= mic_hit_frame:
        # Reveal the path proportionally to frames passed since hit
        reveal_idx = int((frame - mic_hit_frame) * (len(path) / (frames - mic_hit_frame)))
        reveal_idx = min(reveal_idx, len(path))
        line.set_data(path_x[:reveal_idx], path_y[:reveal_idx])
        
    return img, line

ani = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
gif_path = 'acoustic_plate_pathfinding.gif'
ani.save(gif_path, writer='pillow', fps=30)
plt.close(fig)
print(f"Saved simulation to {gif_path}")
