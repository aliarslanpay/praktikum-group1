import matplotlib.pyplot as plt
import numpy as np

# Data extraction
data = {
    'Depth 2': {
        'ENDGAME': [771, 949],
        'MID1': [739, 390],
        'MID2': [1009, 352],
        'default': [968, 297]
    },
    'Depth 3': {
        'ENDGAME': [1000, 825],
        'MID1': [746, 720],
        'MID2': [861, 887],
        'default': [764, 750]
    },
    'Depth 4': {
        'ENDGAME': [843, 974],
        'MID1': [724, 768],
        'MID2': [895, 918],
        'default': [891, 910]
    },
    'Depth 5': {
        'ENDGAME': [952, 872],
        'MID1': [614, 370],
        'MID2': [895, 911],
        'default': [869, 859]
    }
}

data2 = {
    'Thread 2 Depth 3': {
        'midgame1': [1468, 1423],
        'mid2': [1561, 1567],
        'default': [1560, 1339],
        'end': [1709, 1607]
    },
    'Thread 2 Depth 4': {
        'midgame1': [1349, 1558],
        'mid2': [1636, 1723],
        'default': [1586, 843],
        'end': [1773, 1730]
    },
    'Thread 4 Depth 3': {
        'midgame1': [1835, 1531],
        'mid2': [1657, 1657],
        'default': [1382, 685],
        'end': [832, 1779]
    },
    'Thread 4 Depth 4': {
        'midgame1': [1530, 793],
        'mid2': [923, 923],
        'default': [1480, 605],
        'end': [872, 1979]
    },
    'Thread 6 Depth 3': {
        'midgame1': [715, 430],
        'mid2': [698, 1648],
        'default': [445, 1283],
        'end': [1230, 1538]
    },
    'Thread 6 Depth 4': {
        'midgame1': [490, 1089],
        'mid2': [721, 620],
        'default': [565, 1004],
        'end': [520, 543]
    },
    'Thread 8 Depth 3': {
        'midgame1': [261, 500],
        'mid2': [339, 359],
        'default': [298, 245],
        'end': [354, 334]
    },
    'Thread 8 Depth 4': {
        'midgame1': [244, 244],
        'mid2': [384, 295],
        'default': [373, 311],
        'end': [469, 469]
    }
}

# Calculate average values
def calculate_averages(data):
    averages = {}
    for key, value in data.items():
        avg_values = {pos: np.mean(vals) for pos, vals in value.items()}
        averages[key] = np.mean(list(avg_values.values()))
    return averages

avg_data = calculate_averages(data)
avg_data2 = calculate_averages(data2)

# Calculate speedup
speedup = {}
base_depth_3 = avg_data['Depth 3']
base_depth_4 = avg_data['Depth 4']
for key, value in avg_data2.items():
    if 'Depth 3' in key:
        speedup[key] = value / base_depth_3
    elif 'Depth 4' in key:
        speedup[key] = value / base_depth_4

# Plotting
fig, ax = plt.subplots()

threads = [2, 4, 6, 8]
depths = [3, 4]

for depth in depths:
    spd = [speedup.get(f'Thread {t} Depth {depth}', None) for t in threads]
    ax.plot(threads, spd, label=f'Depth {depth}', marker='o')

ax.set_xlabel('Number of Threads')
ax.set_ylabel('Speedup')
ax.set_title('Speedup vs Number of Threads for Depth 3 and 4')
ax.legend()
ax.set_xticks(threads)  # Set x-axis to display only 2, 4, 6, and 8
plt.savefig('speedup.png')
