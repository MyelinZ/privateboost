#!/usr/bin/env python3
"""Regenerate gain_retention figure with cleaner styling."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})

# Data from notebook analysis
features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
retentions = [100.0, 100.0, 100.0, 93.6, 99.7, 100.0, 100.0, 92.1, 100.0, 90.2, 100.0, 100.0, 100.0]

fig, ax = plt.subplots(figsize=(10, 4))

colors = ['#2ecc71' if r >= 99 else '#f39c12' if r >= 90 else '#e74c3c' for r in retentions]

bars = ax.bar(features, retentions, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(100, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylabel('Gain Retention (%)')
ax.set_xlabel('Feature')
ax.set_title('Split Gain Retention: privateboost vs Optimal')
ax.set_ylim(85, 105)
plt.xticks(rotation=45, ha='right')
ax.grid(alpha=0.3, axis='y')

for bar, ret in zip(bars, retentions):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{ret:.0f}%',
            ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('gain_retention.png', dpi=150, bbox_inches='tight')
print("Saved gain_retention.png")
