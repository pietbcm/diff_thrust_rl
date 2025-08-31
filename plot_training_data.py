import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# log_file = "logs/saved_logs/hdot_phi_20_1_N_2000000/events.out.tfevents.1756468421.pietbcm-SAGA-III.18996.0"
# log_file = "logs/saved_logs/hdot_phi_20_3_N_2000000/events.out.tfevents.1756472908.pietbcm-SAGA-III.24947.0"
# log_file = "logs/engine_psidot_hdot/PPO_51/events.out.tfevents.1756493061.pietbcm-SAGA-III.48349.0"

log_files = [
    ("logs/engine_psidot_hdot/PPO_59/events.out.tfevents.1756645848.pietbcm-SAGA-III.25061.0"),
    ("logs/engine_psidot_hdot/PPO_60/events.out.tfevents.1756648576.pietbcm-SAGA-III.29405.0"),
    ("logs/engine_phi_hdot/PPO_1/events.out.tfevents.1756651248.pietbcm-SAGA-III.33378.0"),
    # ("logs/saved_logs/hdot_phi_20_1_N_2000000/events.out.tfevents.1756468421.pietbcm-SAGA-III.18996.0"),
    # ("logs/saved_logs/hdot_phi_20_3_N_2000000/events.out.tfevents.1756472908.pietbcm-SAGA-III.24947.0"),
    # (""),
]

legend_names = ["w_phi = 0.1", "w_phi = 0.2", "w_phi = 0.5"]

tags = ["episode/base_hdot_rew",
        # "episode/base_psidot_rew",
        "episode/base_phi_rew",
        "episode/total_rew"]
window = 50   # moving average window size

def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w

fig, axes = plt.subplots(1, len(tags), figsize=(15, 4))

for log_file in log_files:
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()

    for i, tag in enumerate(tags):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        if len(values) >= window:
            values = moving_average(values, window)
            steps = steps[window-1:]

        axes[i].plot(steps, values)

for i, tag in enumerate(tags):
    axes[i].set_title(tag, fontsize=12)
    axes[i].set_xlabel("Step")
    axes[i].set_ylabel("Value")
    axes[i].grid(True)

    if i == 0: axes[i].legend(legend_names)

plt.tight_layout()
plt.show()