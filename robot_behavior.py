import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import math
import random
import os

os.makedirs('output', exist_ok=True)

def run_auto_quantum_knapsack(items, target_capacity, sim_index, shots=4000):
    # -------------------------------------------------------------------------
    # 0. PARSE THE ITEMS DICTIONARY (LABEL -> WEIGHT)
    # -------------------------------------------------------------------------
    labels = list(items.keys())
    weights = list(items.values())
    total_q = len(items)
    grid_states = 2**total_q
    
    print(f"--- SOLVING QUANTUM KNAPSACK / SUBSET SUM ---")
    print(f"Inventory (Label: Weight): {items}")
    print(f"Target Capacity = {target_capacity}")
    print(f"Allocated {total_q} qubits (Search Space: {grid_states} combinations)")

    # -------------------------------------------------------------------------
    # 1. INITIALIZE CIRCUIT
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2. ORACLE & DIFFUSION LOOP
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 3. MEASUREMENT & DYNAMIC PARSING
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 4. PLOTTING
    # -------------------------------------------------------------------------
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
    img_path = f'output/qiskit_labeled_knapsack_{sim_index}.png'
    fig.savefig(img_path, dpi=100)
    plt.close(fig)
    print(f"Saved quantum output to {img_path}")
    
    # Extract states
    behaviors = []
    offsets = []
    if best_state:
        for idx in range(total_q):
            if best_state[total_q - 1 - idx] == '1':
                if 'behavior' in labels[idx]:
                    behaviors.append(weights[idx])
                elif 'target_offset' in labels[idx]:
                    offsets.append(weights[idx])
                    
    return behaviors, offsets

def make_auto_inventory(n_items=5, min_weight=1, max_weight=50):
    chosen_labels = [f"behavior_{idx+1}" for idx in range(n_items)]
    return {label: random.randint(min_weight, max_weight) for label in chosen_labels}

def make_auto_inventory_offset(n_items=3, min_weight=10, max_weight=40):
    chosen_labels = [f"target_offset_{idx+1}" for idx in range(n_items)]
    return {label: random.randint(min_weight, max_weight) for label in chosen_labels}


def simulate_3d_arm(behaviors, offsets, sim_index):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')

    L1, L2 = 2.0, 1.5
    
    # Use behaviors to dynamically alter space
    avg_behavior = sum(behaviors) / max(len(behaviors), 1)
    
    # Meaningful Variation 1: Target Position depends on behaviors
    t_x = 1.0 + (avg_behavior % 1.5)  # Shifts X coordinate
    t_y = -1.5 + (sum(behaviors) % 3.0) # Shifts Y coordinate
    target_pos = np.array([t_x, t_y, 0.2])
    
    # Meaningful Variation 2: Obstacle height and safe clearance
    obs_h = 0.5 + (sum(behaviors) % 1.5)
    obs_pos = np.array([1.5, 0.5, obs_h])
    safe_clearance = obs_h + 0.5 + (avg_behavior % 1.0) 

    # Meaningful Variation 3: Arm timing/speed depends on offsets
    if not offsets: offsets = [25]
    total_frames = max(20, min(80, sum(offsets)))
    frames = int(total_frames)

    def inverse_kinematics(x, y, z):
        theta1 = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        d = np.sqrt(r**2 + z**2)
        if d > L1 + L2: d = L1 + L2 - 0.01
        cos_theta3 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
        theta3 = np.arccos(np.clip(cos_theta3, -1.0, 1.0))
        theta2 = np.arctan2(z, r) + np.arctan2(L2 * np.sin(theta3), L1 + L2 * np.cos(theta3))
        return theta1, theta2, -theta3

    wp0 = np.array([0.5, 0.0, 3.0])
    wp1 = np.array([1.5, 0.5, safe_clearance])
    wp2 = target_pos

    path = []
    for step in range(frames):
        t = step / max(1, frames - 1)
        if t < 0.5:
            s = t * 2
            pos = wp0 * (1 - s) + wp1 * s
        else:
            s = (t - 0.5) * 2
            pos = wp1 * (1 - s) + wp2 * s
        path.append(pos)

    line, = ax.plot([], [], [], 'o-', color='#00ffcc', lw=4, markersize=8)
    ax.scatter(*obs_pos, color='#ff0033', s=200, label=f'Obstacle (H={obs_h:.1f})')
    ax.scatter(*target_pos, color='#ffcc00', s=150, marker='*', label=f'Target (B={avg_behavior:.1f})')

    ax.set_xlim([-1, 3])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 4])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f"Sim {sim_index} | Frames: {frames} | Clearance: {safe_clearance:.1f}", color='white')
    ax.legend(facecolor='#222222', labelcolor='white')

    def update(frame):
        pos = path[frame]
        th1, th2, th3 = inverse_kinematics(*pos)
        x1 = L1 * np.cos(th2) * np.cos(th1)
        y1 = L1 * np.cos(th2) * np.sin(th1)
        z1 = L1 * np.sin(th2)
        x2 = x1 + L2 * np.cos(th2 + th3) * np.cos(th1)
        y2 = y1 + L2 * np.cos(th2 + th3) * np.sin(th1)
        z2 = z1 + L2 * np.sin(th2 + th3)
        line.set_data(np.array([[0, x1, x2], [0, y1, y2]]))
        line.set_3d_properties(np.array([0, z1, z2]))
        return line,

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    gif_path = f'output/arm_sim_{sim_index}.gif'
    ani.save(gif_path, writer='pillow')
    plt.close(fig)
    print(f"Saved sim to {gif_path}")


# Run 3 distinct simulations iteratively
num_simulations = 32
for i in range(1, num_simulations + 1):
    print(f"\n{'='*40}\nSTARTING SIMULATION {i}\n{'='*40}")
    
    # 8 qubits total to avoid circuit compilation timeouts in the loop
    inv = make_auto_inventory(n_items=5)
    off = make_auto_inventory_offset(n_items=3)
    combined = inv | off
    
    h_keys = random.sample(list(inv.keys()), 2) + random.sample(list(off.keys()), 1)
    target_w = sum(combined[k] for k in h_keys)
    
    print(f"[AUTO-GEN] Planted a valid combination: {h_keys} which sums to {target_w}")
    b, o = run_auto_quantum_knapsack(combined, target_w, i)
    print(f"Quantum Extracted -> Behaviors: {b}, Offsets: {o}")
    
    simulate_3d_arm(b, o, i)
