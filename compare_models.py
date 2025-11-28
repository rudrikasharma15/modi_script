import json
import matplotlib.pyplot as plt
import numpy as np

print("Loading BEFORE analysis...")
with open('layer_analysis/layer_analysis_report.json', 'r') as f:
    before = json.load(f)

print("Loading AFTER analysis...")
with open('layer_analysis_AFTER_FIX/layer_analysis_report.json', 'r') as f:
    after = json.load(f)

key_layers = ['Layer_0_Conv', 'Layer_1_Conv', 'Layer_7_Conv', 'Layer_19_Conv', 'Layer_21_C2f']

before_scores = []
after_scores = []
layer_names = []

for layer in key_layers:
    if layer in before['top_failing_layers'] and layer in after['top_failing_layers']:
        before_scores.append(before['top_failing_layers'][layer]['problem_score'])
        after_scores.append(after['top_failing_layers'][layer]['problem_score'])
        layer_names.append(layer)

before_diffs = []
after_diffs = []
for layer in layer_names:
    before_pct = before['top_failing_layers'][layer]['mean_activation']['percentage_change']
    after_pct = after['top_failing_layers'][layer]['mean_activation']['percentage_change']
    before_diffs.append(before_pct)
    after_diffs.append(after_pct)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Layer Analysis: Before vs After Optimization', fontsize=16, fontweight='bold')

x = np.arange(len(layer_names))
width = 0.35

ax1 = axes[0, 0]
bars1 = ax1.bar(x - width/2, before_scores, width, label='Before Fix', alpha=0.8)
bars2 = ax1.bar(x + width/2, after_scores, width, label='After Fix', alpha=0.8)
ax1.set_xlabel('Layer Name')
ax1.set_ylabel('Problem Score')
ax1.set_title('Problem Scores: Before vs After')
ax1.set_xticks(x)
ax1.set_xticklabels(layer_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

ax2 = axes[0, 1]
bars1 = ax2.bar(x - width/2, before_diffs, width, label='Before Fix', alpha=0.8)
bars2 = ax2.bar(x + width/2, after_diffs, width, label='After Fix', alpha=0.8)
ax2.set_xlabel('Layer Name')
ax2.set_ylabel('Activation Difference (%)')
ax2.set_title('Activation Differences: Before vs After')
ax2.set_xticks(x)
ax2.set_xticklabels(layer_names, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

improvements = []
for i in range(len(before_scores)):
    improvement = ((before_scores[i] - after_scores[i]) / before_scores[i]) * 100
    improvements.append(improvement)

ax3 = axes[1, 0]
colors = ['green' if imp > 0 else 'red' for imp in improvements]
bars = ax3.bar(layer_names, improvements, color=colors, alpha=0.8)
ax3.set_xlabel('Layer Name')
ax3.set_ylabel('Improvement (%)')
ax3.set_title('Problem Score Improvement (%)')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax3.grid(axis='y', alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax4 = axes[1, 1]
ax4.axis('off')

plt.tight_layout()
plt.savefig('BEFORE_AFTER_COMPARISON.png', dpi=300, bbox_inches='tight')
print("Comparison saved: BEFORE_AFTER_COMPARISON.png")
