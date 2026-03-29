
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import math
import random
import os
import csv
import zipfile

os.makedirs('output', exist_ok=True)
os.makedirs('output/data', exist_ok=True)
os.makedirs('output/sims', exist_ok=True)
os.makedirs('output/charts', exist_ok=True)

def run_auto_quantum_knapsack(items, target_capacity, sim_index, shots=2000):
    labels = list(items.keys())
    weights = list(items.values())
    total_q = len(items)
    grid_states = 2**total_q
    
    print(f"--- SOLVING QUANTUM KNAPSACK / SUBSET SUM ---")
    print(f"Inventory (Label: Weight): {items}")
    print(f"Target Capacity = {target_capacity}")
    print(f"Allocated {total_q} qubits (Search Space: {grid_states} combinations)")

    qc = QuantumCircuit(total_q + 1, total_q)
    qc.x(total_q)
    qc.h(total_q)
    qc.h(range(total_q))
    qc.barrier()

    targets = []
    for state in range(grid_states):
        current_sum = sum(weights[i] for i in range(total_q) if (state & (1 << i)))
        if current_sum == target_capacity:
            targets.append(state)

    if len(targets) > 0:
        iterations = int((math.pi / 4.0) * math.sqrt(grid_states / len(targets)))
        if iterations == 0: iterations = 1
    else:
        iterations = 1 
        
    print(f"Optimal wave iterations (bounces): {iterations}")

    for step in range(iterations):
        for state in targets:
            for idx in range(total_q):
                if (state & (1 << idx)) == 0: qc.x(idx)
            qc.mcx(list(range(total_q)), total_q)
            for idx in range(total_q):
                if (state & (1 << idx)) == 0: qc.x(idx)
        qc.barrier()

        qc.h(range(total_q))
        qc.x(range(total_q))
        qc.h(total_q - 1)
        if total_q > 1:
            qc.mcx(list(range(total_q - 1)), total_q - 1)
        qc.h(total_q - 1)
        qc.x(range(total_q))
        qc.h(range(total_q))
        qc.barrier()

    qc.measure(range(total_q), range(total_q))

    sim = AerSimulator()
    compiled_qc = transpile(qc, sim)
    result = sim.run(compiled_qc, shots=shots).result()
    raw_counts = result.get_counts()

    noise_floor = shots / grid_states
    threshold = noise_floor * 2

    formatted_counts = {}
    best_state = None
    max_count = 0
    
    for bitstring, count in raw_counts.items():
        if count > threshold:
            selected_labels = []
            for idx in range(total_q):
                if bitstring[total_q - 1 - idx] == '1':
                    selected_labels.append(f"{labels[idx]} ({weights[idx]})")
            
            label_text = "Empty" if not selected_labels else "\n+\n".join(selected_labels)
            formatted_counts[label_text] = count
            if count > max_count:
                max_count = count
                best_state = bitstring

    if not formatted_counts:
        print(f"\n=> No valid subsets found! The signal is flat noise.")
        sorted_noise = sorted(raw_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for bitstring, count in sorted_noise:
            selected_labels = []
            for idx in range(total_q):
                if bitstring[total_q - 1 - idx] == '1':
                    selected_labels.append(f"{labels[idx]} ({weights[idx]})")
            label_text = "Noise: Empty" if not selected_labels else "Noise:\n+" + "\n+".join(selected_labels)
            formatted_counts[label_text] = count
    else:
        print(f"\n=> Significant item combinations found hitting Target={target_capacity}:")
        for k, v in formatted_counts.items():
            clean_k = k.replace('\n', ' ')
            print(f"   {clean_k} -> {v} shots")

    fig, ax = plt.subplots(figsize=(14, 8))
    plot_histogram(formatted_counts, ax=ax, color='#ff00cc')
    ax.set_title(f"Quantum Item Selection Sim {sim_index} (Target Weight: {target_capacity})", fontsize=14)
    ax.set_xlabel("Valid Label + Weight Combinations")
    ax.set_ylabel("Measurement Amplitude (Shots)")
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')
    ax.xaxis.label.set_color('lightgray')
    ax.yaxis.label.set_color('lightgray')
    ax.title.set_color('white')
    ax.tick_params(colors='lightgray', labelsize=8)

    plt.tight_layout()
    img_path = f'output/charts/qiskit_labeled_knapsack_{sim_index}.png'
    fig.savefig(img_path, dpi=100)
    plt.close(fig)
    print(f"Saved quantum output to {img_path}")

    behaviors, offsets = [], []
    if best_state:
        for idx in range(total_q):
            if best_state[total_q - 1 - idx] == '1':
                if 'behavior' in labels[idx]:
                    behaviors.append(weights[idx])
                elif 'target_offset' in labels[idx]:
                    offsets.append(weights[idx])
                    
    return behaviors, offsets

def make_auto_inventory(n_items=5, min_weight=1, max_weight=50):
    return {f"behavior_{idx+1}": random.randint(min_weight, max_weight) for idx in range(n_items)}

def make_auto_inventory_offset(n_items=3, min_weight=10, max_weight=40):
    return {f"target_offset_{idx+1}": random.randint(min_weight, max_weight) for idx in range(n_items)}

def draw_cube(ax, bottom_center, size, color='#ff0033'):
    x, y, z = bottom_center
    s = size / 2.0
    h = size
    v = np.array([
        [x-s, y-s, z], [x+s, y-s, z], [x+s, y+s, z], [x-s, y+s, z],
        [x-s, y-s, z+h], [x+s, y-s, z+h], [x+s, y+s, z+h], [x-s, y+s, z+h]
    ])
    faces = [
        [v[0], v[1], v[5], v[4]], [v[1], v[2], v[6], v[5]],
        [v[2], v[3], v[7], v[6]], [v[3], v[0], v[4], v[7]],
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='black', alpha=0.7))

def simulate_3d_arm_true_avoidance(behaviors, offsets, sim_index):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')

    L1, L2 = 2.0, 1.5
    avg_behavior = sum(behaviors) / max(len(behaviors), 1)
    
    p_x = 1.0 + (avg_behavior % 1.0)
    p_y = -0.5 - (sum(behaviors) % 1.0)
    pickup_pos = np.array([p_x, p_y, 0.2])
    
    d_x = -1.0 - (sum(behaviors) % 0.8)
    d_y = 1.0 + (avg_behavior % 1.0)
    drop_pos = np.array([d_x, d_y, 0.2])
    
    num_obstacles = max(2, min(5, len(offsets) + (sum(behaviors) % 3)))
    obstacles = []
    max_obs_height = 0
    
    for i in range(int(num_obstacles)):
        ox = random.uniform(-0.8, 0.8)
        oy = random.uniform(-0.8, 0.8)
        osize = 0.5 + (random.random() * 0.7)
        
        obstacles.append({'pos': np.array([ox, oy, 0]), 'size': osize})
        draw_cube(ax, [ox, oy, 0], osize, color='#ff0033')
        if osize > max_obs_height:
            max_obs_height = osize

    safe_z = max_obs_height + 1.2

    if not offsets: offsets = [30]
    frames = int(max(40, min(120, sum(offsets) * 1.5)))

    def inverse_kinematics(x, y, z):
        theta1 = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        d = np.sqrt(r**2 + z**2)
        if d >= L1 + L2: 
            d = L1 + L2 - 0.001
        cos_theta3 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        theta3 = -np.arccos(cos_theta3) 
        theta2 = np.arctan2(z, r) + np.arctan2(L2 * np.sin(-theta3), L1 + L2 * np.cos(-theta3))
        return theta1, theta2, theta3

    wp_start = np.array([0.0, 0.0, 3.0])
    wp_pickup = pickup_pos
    wp_pickup_high = np.array([pickup_pos[0], pickup_pos[1], safe_z])
    wp_drop_high = np.array([drop_pos[0], drop_pos[1], safe_z])
    wp_drop = drop_pos

    path, obj_path = [], []
    for step in range(frames):
        t = step / (frames - 1)
        if t < 0.15:
            s = t / 0.15
            arm_pos = wp_start * (1 - s) + wp_pickup * s
            obj_pos = wp_pickup 
        elif t < 0.35:
            s = (t - 0.15) / 0.20
            arm_pos = wp_pickup * (1 - s) + wp_pickup_high * s
            obj_pos = arm_pos 
        elif t < 0.75:
            s = (t - 0.35) / 0.40
            arm_pos = wp_pickup_high * (1 - s) + wp_drop_high * s
            arm_pos[2] += np.sin(s * np.pi) * 0.5 
            obj_pos = arm_pos
        else:
            s = (t - 0.75) / 0.25
            arm_pos = wp_drop_high * (1 - s) + wp_drop * s
            obj_pos = arm_pos
        path.append(arm_pos)
        obj_path.append(obj_pos)

    # -------------------------------------------------------------------------
    # GENERATE CSV WITH KINEMATIC LOGGING
    # -------------------------------------------------------------------------
    csv_path = f'output/data/kinematic_log_sim_{sim_index}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'arm_x', 'arm_y', 'arm_z', 'theta1_yaw', 'theta2_shoulder', 'theta3_elbow'])
        for step in range(frames):
            pos = path[step]
            t1, t2, t3 = inverse_kinematics(*pos)
            writer.writerow([step, round(pos[0],4), round(pos[1],4), round(pos[2],4), round(t1,4), round(t2,4), round(t3,4)])
    
    print(f"Saved kinematic trajectory log to {csv_path}")

    # -------------------------------------------------------------------------
    # RENDER 3D VISUALIZATION
    # -------------------------------------------------------------------------
    line, = ax.plot([], [], [], 'o-', color='#00ffcc', lw=5, markersize=7, zorder=10)
    payload_marker, = ax.plot([], [], [], 'o', color='#ff66ff', markersize=10, label='Payload', zorder=11)
    
    xx, yy = np.meshgrid([-2, 3], [-2, 3])
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
    
    ax.scatter(*drop_pos, color='#ffcc00', s=150, marker='s', alpha=0.8, label='Drop Zone')
    ax.scatter(*pickup_pos, color='#00ffcc', s=50, marker='x', label='Pickup')

    ax.set_xlim([-2, 3])
    ax.set_ylim([-2, 3])
    ax.set_zlim([0, 4])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"Sim {sim_index} | {int(num_obstacles)} Cubes | Z-Clearance: {safe_z:.2f}", color='white')
    ax.legend(facecolor='#222222', labelcolor='white')

    def update(frame):
        pos = path[frame]
        th1, th2, th3 = inverse_kinematics(*pos)
        x0, y0, z0 = 0, 0, 0
        x1 = L1 * np.cos(th2) * np.cos(th1)
        y1 = L1 * np.cos(th2) * np.sin(th1)
        z1 = L1 * np.sin(th2)
        x2 = x1 + L2 * np.cos(th2 + th3) * np.cos(th1)
        y2 = y1 + L2 * np.cos(th2 + th3) * np.sin(th1)
        z2 = z1 + L2 * np.sin(th2 + th3)
        
        line.set_data(np.array([[x0, x1, x2], [y0, y1, y2]]))
        line.set_3d_properties(np.array([z0, z1, z2]))
        
        op = obj_path[frame]
        payload_marker.set_data(np.array([op[0]]), np.array([op[1]]))
        payload_marker.set_3d_properties(np.array([op[2]]))
        return line, payload_marker

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    gif_path = f'output/sims/arm_sim_true_avoidance_{sim_index}.gif'
    ani.save(gif_path, writer='pillow')
    plt.close(fig)
    print(f"Saved sim visual to {gif_path}")

# Run 4 distinct simulations iteratively
num_simulations = 4
for i in range(1, num_simulations + 1):
    print(f"\n{'='*40}\nSTARTING SIMULATION {i}\n{'='*40}")
    inv = make_auto_inventory(n_items=5)
    off = make_auto_inventory_offset(n_items=3)
    combined = inv | off
    
    h_keys = random.sample(list(inv.keys()), 2) + random.sample(list(off.keys()), 1)
    target_w = sum(combined[k] for k in h_keys)
    
    print(f"[AUTO-GEN] Planted a valid combination: {h_keys} which sums to {target_w}")
    b, o = run_auto_quantum_knapsack(combined, target_w, i)
    print(f"Extracted Behaviors: {b}, Offsets: {o}")
    simulate_3d_arm_true_avoidance(b, o, i)

# Package all outputs into a ZIP file for the user
import shutil
zip_filename = 'output/Quantum_Kinematics_Bundle.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for root, dirs, files in os.walk('output'):
        for file in files:
            if file != 'Quantum_Kinematics_Bundle.zip':
                file_path = os.path.join(root, file)
                # Keep directory structure inside zip
                arcname = os.path.relpath(file_path, 'output')
                zipf.write(file_path, arcname)

print(f"\nSuccessfully bundled all outputs into {zip_filename}")
