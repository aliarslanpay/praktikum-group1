import matplotlib.pyplot as plt

# 数据提取
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

depths = list(data.keys())
eval_rates_avg = {key: [] for key in data['Depth 2'].keys()}

for depth in depths:
    for key in data[depth]:
        avg_rate = sum(data[depth][key]) / 2
        eval_rates_avg[key].append(avg_rate)

plt.figure(figsize=(10, 6))

for scenario, rates in eval_rates_avg.items():
    plt.plot(depths, rates, marker='o', label=scenario)

plt.xlabel('Search Depth')
plt.ylabel('Average EvalRate (k/s)')
plt.title('Group 1: Average EvalRate vs Search Depth')
plt.legend()
plt.grid(axis='y')
plt.savefig('evalrate_vs_depth.pdf')
