# %%
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# %%
vanilla = pd.read_json('Data/Eval/MathVMini/GLM-4.1V-9B-Thinking/MathVMini-GLM-4.1V-9B-Thinking_0.0_all_eval_vote_num1_0.0.json')
steer = pd.read_json("Data/Eval/MathVMini/GLM-4.1V-9B-Thinking/MathVMini-GLM-4.1V-9B-Thinking_0.0_all_eval_vote_num1_1.5.json")

# %%
vanilla.avg_llm_reasoning_token_num.mean()

# %%
phrases = ["can't see the image", "i can't see it", "don't have access to",
           "don't have it", "don't have the image", "can't see the actual image",
           "don't have the actual image", "can't see the picture", "don't have the picture"]

def join_reasoning(x):
    if isinstance(x, list):
        return "\n".join(str(s) for s in x)
    return str(x)

text = vanilla["llm_reasoning"].apply(join_reasoning).str.lower()

any_hit = text.apply(lambda t: any(p in t for p in phrases))
phrase_hits = {p: text.str.contains(re.escape(p), regex=True).sum() for p in phrases}

print("rows with any hit:", int(any_hit.sum()), "/", len(vanilla))
for p, c in phrase_hits.items():
    print(f"{p!r}: {c}")

matches = vanilla.loc[any_hit, ["id", "llm_reasoning"]]
matches.head()


# %%
text_steer = steer["llm_reasoning"].apply(join_reasoning).str.lower()

any_hit_steer = text_steer.apply(lambda t: any(p in t for p in phrases))
phrase_hits_steer = {p: text_steer.str.contains(re.escape(p), regex=True).sum() for p in phrases}

print("rows with any hit:", int(any_hit_steer.sum()), "/", len(steer))
for p, c in phrase_hits_steer.items():
    print(f"{p!r}: {c}")

matches_steer = steer.loc[any_hit_steer, ["id", "llm_reasoning"]]
matches_steer.head()


# %%


font_path = "Assets/Times New Roman Bold.ttf"
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)

plt.rcParams["font.family"] = font_prop.get_name()

x = np.arange(len(phrases))
van_counts = [phrase_hits[p] for p in phrases]
ste_counts = [phrase_hits_steer[p] for p in phrases]
width = 0.4

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, van_counts, width, label="Vanilla")
bars2 = ax.bar(x + width/2, ste_counts, width, label="Ours")

ax.set_xticks(x, phrases, rotation=45, ha="right")
ax.set_ylabel("Count")
ax.set_title("Vanilla vs Ours (Qwen3-VL-4B-Thinking)")
ax.legend()

try:
    ax.bar_label(bars1, padding=3, fontsize=8)
    ax.bar_label(bars2, padding=3, fontsize=8)
except AttributeError:
    for bars in (bars1, bars2):
        for b in bars:
            ax.annotate(f"{int(b.get_height())}",
                        (b.get_x() + b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0,3),
                        textcoords="offset points")


# %%
def to_scalar(x):
    if isinstance(x, list):
        if len(x) == 0:
            return np.nan
        x = x[0]
    return pd.to_numeric(x, errors="coerce")

text = vanilla["llm_reasoning"].apply(join_reasoning).str.lower()
has_phrase = text.apply(lambda t: any(p in t for p in phrases))

key = "id" if "id" in vanilla.columns and "id" in steer.columns else "question_id"

v = vanilla[[key, "avg_llm_reasoning_token_num"]].copy()
s = steer[[key, "avg_llm_reasoning_token_num"]].copy()

v["v_avg"] = v["avg_llm_reasoning_token_num"].apply(to_scalar)
s["s_avg"] = s["avg_llm_reasoning_token_num"].apply(to_scalar)

phrase_flag = vanilla[[key]].copy()
phrase_flag["has_phrase"] = has_phrase.values

m = (
    v[[key, "v_avg"]]
    .merge(s[[key, "s_avg"]], on=key, how="inner")
    .merge(phrase_flag, on=key, how="left")
)

m["diff"] = m["s_avg"] - m["v_avg"]

def summary(df, name):
    df = df.dropna(subset=["v_avg", "s_avg"])
    n = len(df)
    reduced = (df["diff"] < 0).sum()
    print(f"\n{name}")
    print(f"  rows: {n}")
    print(f"  reduced (steer < vanilla): {reduced} / {n} = {reduced/n:.2%}")
    print(f"  mean diff (steer - vanilla): {df['diff'].mean():.2f}")
    print(f"  median diff: {df['diff'].median():.2f}")
    print(f"  vanilla mean: {df['v_avg'].mean():.2f}, steer mean: {df['s_avg'].mean():.2f}")

summary(m[m["has_phrase"] == True],  "Group A: vanilla has phrases")
summary(m[m["has_phrase"] == False], "Group B: vanilla no phrases")

# %%
def to_scalar(x):
    if isinstance(x, list):
        if len(x) == 0:
            return np.nan
        x = x[0]
    return pd.to_numeric(x, errors="coerce")

text = vanilla["llm_reasoning"].apply(join_reasoning).str.lower()
has_phrase = text.apply(lambda t: any(p in t for p in phrases))

key = "id" if "id" in vanilla.columns and "id" in steer.columns else "question_id"

v = vanilla[[key, "avg_llm_reasoning_token_num"]].copy()
s = steer[[key, "avg_llm_reasoning_token_num"]].copy()

v[key] = v[key].astype(str)
s[key] = s[key].astype(str)

v["v_avg"] = v["avg_llm_reasoning_token_num"].apply(to_scalar)
s["s_avg"] = s["avg_llm_reasoning_token_num"].apply(to_scalar)

phrase_flag = vanilla[[key]].copy()
phrase_flag[key] = phrase_flag[key].astype(str)
phrase_flag["has_phrase"] = has_phrase.values

m = (
    v[[key, "v_avg"]]
    .merge(s[[key, "s_avg"]], on=key, how="inner")
    .merge(phrase_flag, on=key, how="left")
)

m["diff"] = m["s_avg"] - m["v_avg"]
m["pct_change"] = m["diff"] / (m["v_avg"] + 1e-8)

def summarize_group(df, name):
    df = df.dropna(subset=["v_avg", "s_avg"])
    n = len(df)
    print(f"\n{name}")
    print(f"  rows: {n}")
    print(f"  vanilla mean: {df['v_avg'].mean():.2f}")
    print(f"  steer mean:   {df['s_avg'].mean():.2f}")
    print(f"  mean diff (steer - vanilla): {df['diff'].mean():.2f}")
    print(f"  median diff: {df['diff'].median():.2f}")
    print(f"  mean % change: {df['pct_change'].mean():.2%}")
    print(f"  reduced (steer < vanilla): {(df['diff']<0).sum()} / {n} = {(df['diff']<0).mean():.2%}")

summarize_group(m[m["has_phrase"] == True],  "Group A: has phrase")
summarize_group(m[m["has_phrase"] == False], "Group B: no phrase")


# %%
groups = ["w/ Phrases", "w/o Phrases"]
van_means = [10085.29, 9651.58]
ste_means = [7036.81, 6991.69]

x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
bars1 = ax.bar(x - width/2, van_means, width, label="Vanilla", color="#4C78A8")
bars2 = ax.bar(x + width/2, ste_means, width, label="Ours",   color="#F58518")

ax.set_xticks(x, groups)
ax.set_ylabel("avg_llm_reasoning_token_num")
ax.set_title("Mean Token Length by Group")
ax.legend(fontsize=8)
ax.set_ylim(0, max(van_means + ste_means) * 1.1)
ax.bar_label(bars1, padding=2, fontsize=8)
ax.bar_label(bars2, padding=2, fontsize=8)

plt.tight_layout()
plt.show()

# %%
count_sum = [sum(van_counts), sum(ste_counts)]
category = ["Vanilla", "Ours"]

plt.figure(figsize=(4,4), dpi=300)
bars = plt.bar(category, count_sum, color=["#4C78A8", "#F58518"])
plt.bar_label(bars, padding=3)
plt.ylim(0, max(count_sum) * 1.15)
#plt.title("Total No-Image Phrase (Qwen3-VL-8B-Thinking)")
plt.title("Total No-Image Phrase (GLM-4.1V-9B-Thinking)")
plt.show()


# %%
groups = ["w/ \"No-Image\" Phrases", "w/o \"No-Image\" Phrases"]
# reduced_rate = [0.8333, 0.7849]          # 83.33%, 78.49%
mean_diff = [753.70, 293.91]
#median_diff = [2831.50, 2538.00]

fig, axes = plt.subplots(figsize=(4, 4), dpi=300)


"""
# Reduction Rate
axes[0].bar(groups, reduced_rate, color=["#4C78A8", "#F58518"])
axes[0].set_ylim(0, 1)
axes[0].set_title("Reduction rate")
axes[0].set_ylabel("steer < vanilla")
for i, v in enumerate(reduced_rate):
    axes[0].text(i, v+0.02, f"{v*100:.2f}%", ha="center")
"""
    
# Mean Diff
axes.bar(groups, mean_diff, color=["#4C78A8", "#F58518"])
#axes.set_title("Average #Token Reduction (Qwen3-VL-2B-Thinking)")
axes.set_title("Average #Token Reduction (GLM-4.1V-9B-Thinking)")
for i, v in enumerate(mean_diff):
    axes.text(i, v+30, f"{v:.2f}", ha="center")
plt.ylim(0, max(mean_diff) * 1.1)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

bars1 = axes[0].bar(category, count_sum, color=["#4C78A8", "#F58518"])
axes[0].bar_label(bars1, padding=3, fontsize=16)
axes[0].set_ylim(0, max(count_sum) * 1.15)
axes[0].tick_params(axis="both", labelsize=12)
axes[0].set_title("Total No-Image Phrase Frequency", fontsize=16)

bars2 = axes[1].bar(groups, mean_diff, color=["#4C78A8", "#F58518"])
axes[1].bar_label(bars2, padding=3, fontsize=16)
axes[1].tick_params(axis="both", labelsize=12)
axes[1].set_title("Average #Token Reduction after Steering", fontsize=16)
axes[1].set_ylim(0, max(mean_diff) * 1.2)

axes[0].tick_params(axis="x", labelsize=16)
axes[1].tick_params(axis="x", labelsize=16)

plt.tight_layout()
plt.show()


# %%
# 4b
#alphas = [0.0, 0.8, 0.95, 1.1, 1.25, 1.4]
#acc1 = [0.64, 0.64, 0.62, 0.6466666666666666, 0.6066666666666667, 0.6333333333333333]
#tokens1 = [6459.206666666667, 4964.826666666667, 4619.74, 4405.326666666667, 4352.893333333333, 4185.3133333333335]

#2b
#alphas = [0.0, 0.35, 0.5, 0.65, 0.8, 1.1, 1.4]
#acc1 = [0.52, 0.54, 0.56, 0.5333333333333333, 0.5333333333333333, 0.52, 0.5066666666666667]
#tokens1 = [8119.3, 6787.8133333333335, 6438.6866666666665, 6292.693333333334, 6290.7266666666665, 5746.82, 5106.82]

# glm
#alphas = [0.0, 0.35, 0.5, 0.65, 0.8, 1.1]
#acc1 = [0.7333333333333333, 0.7, 0.74, 0.72, 0.72, 0.7133333333333334]
#tokens1 = [1720.6866666666667, 1722.16, 1629.38, 1654.6133333333332, 1676.98, 1542.4333333333334]

alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
acc1 = [0.7333333333333333, 0.7, 0.72, 0.7333333333333333, 0.72, 0.7133333333333334, 0.6533333333333333]
tokens1 = [1720.6866666666667, 1722.16, 1598.44, 1548.9333333333334, 1477.7533333333333, 1312.58, 1167.4333333333334]

#8b
#alphas = [0.0, 0.5, 0.8, 0.95, 1.1, 1.4]
#acc1 = [0.6533333333333333, 0.6266666666666667, 0.6066666666666667, 0.6666666666666666, 0.6133333333333333, 0.60]
#tokens1 = [4947.073333333334, 4165.3133333333335, 3474.72, 3709.786666666667, 3262.233333333333, 3005.673333333333]

x = np.arange(len(alphas))

fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)
ax2 = ax1.twinx()
ax2.set_ylim(1000, 1850) 
#ax2.set_ylim(0, 6000) 

bar = ax2.bar(x, tokens1, width=0.6, color="#4C78A8", label='Avg Reasoning Tokens')
line = ax1.plot(x, acc1, color="#F58518", marker='o', linewidth=2, label='Accuracy')

ax1.set_zorder(3)
ax1.patch.set_visible(False)
ax1.set_xlabel('Adaptive alpha', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax2.set_ylabel('Avg Reasoning Tokens', fontsize=14)

ax1.set_xticks(x)
ax1.set_xticklabels([str(a) for a in alphas])
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

lines = line + [bar]
labels = [l.get_label() for l in line] + [bar.get_label()]
ax1.legend(lines, labels, loc='upper right')

ax1.set_title("GLM-4.1V-9B-Thinking", fontsize=16)
#ax1.set_title("Qwen3-VL-8B-Thinking", fontsize=16)
fig.tight_layout()
plt.show()

# %%
#4b
"""
labels = ["All Layers", "Layer 12"]
acc2 = [0.6333333333333333, 0.6333333333333333]
tokens2 = [4406.36, 5603.266666666666]

ref_acc = 0.6466666666666666
ref_tokens = 4183.56
"""

#2b
"""
labels = ["All Layers @ alpha=0.1637", "Layer 11 @ alpha=0.8660"]
acc2 = [0.54, 0.54]
tokens2 = [7148.64, 7044.953333333333]

ref_acc = 0.56
ref_tokens = 6438.6866666666665
"""

#8b
"""
labels = ["All Layers @ alpha=0.3540", "Layer 12 @ alpha=2.1243"]
acc2 = [0.64, 0.6533333333333333]
tokens2 = [3652.3, 3729.1133333333332]

ref_acc = 0.6666666666666666
ref_tokens = 3709.786666666667
"""

#glm

labels = ["All Layers @ alpha=0.5303", "Layer 9 @ alpha=3.3541"]
acc2 = [0.7266666666666667, 0.6933333333333334]
tokens2 = [1546.74, 1512.4533333333334]

ref_acc = 0.7333333333333333
ref_tokens = 1548.9333333333334


x = np.arange(len(labels))
w = 0.35

c_tok = "#2C5C8A"  # deep orange
c_acc = "#D55E00"    # deep blue

fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)
ax2 = ax1.twinx()

# bars
bar_acc = ax1.bar(x - w/2, acc2, width=w, color="#F58518", label="Accuracy", zorder=1)
bar_tok = ax2.bar(x + w/2, tokens2, width=w, color="#4C78A8", label="Avg Reasoning Tokens", zorder=1)

# labels on bars
ax1.bar_label(bar_acc, fmt="%.3f", padding=4, fontsize=14)
ax2.bar_label(bar_tok, fmt="%.1f", padding=4, fontsize=14)

# dashed reference lines (on top)
ax1.axhline(ref_acc, color=c_acc, linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Ref Accuracy")
ax2.axhline(ref_tokens, color=c_tok, linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Ref Tokens")

# ax1.set_xlabel("Setting", fontsize=14)
ax1.set_ylabel("Accuracy", fontsize=14)
ax2.set_ylabel("Avg Reasoning Tokens", fontsize=14)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=14)

# headroom
ax1.set_ylim(0.65, 0.77)
#ax1.set_ylim(0, max(max(acc2), ref_acc) * 1.5)
ax2.set_ylim(1200, 1800)

ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

# legend
handles = [bar_acc, bar_tok, ax1.lines[0], ax2.lines[0]]
labels = ["Accuracy", "Avg Reasoning Tokens", "Steer(7-11) Accuracy @ alpha = 1.5", "Steer(7-11) Tokens @ alpha = 1.5"]
ax1.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 1.02), framealpha=0.9)

ax2.set_zorder(ax1.get_zorder() + 1)
ax2.patch.set_visible(False)
ax1.patch.set_visible(False)

ax1.set_title("GLM-4.1V-9B-Thinking", fontsize=16)
# ax1.set_title("Qwen3-VL-2B-Thinking", fontsize=16)
fig.tight_layout()
plt.show()


# %%
# --------- subplot 1 data ---------
alphas = [0.0, 0.5, 0.8, 0.95, 1.1, 1.4]
acc1 = [0.64, 0.6533333333333333, 0.6266666666666667, 0.6533333333333333, 0.5933333333333334, 0.5866666666666667]
tokens1 = [6459.206666666667, 4925.113333333334, 4191.98, 3938.66, 3405.34, 3000.32]

# --------- subplot 2 data ---------
labels = ["All Layers @ alpha = 0.3540", "Layer 12 @ alpha = 2.1243"]
acc2 = [0.6333333333333333, 0.6466666666666666]
tokens2 = [4192.66, 5603.266666666666]

ref_acc = 0.6533333333333333
ref_tokens = 3938.66


# %%
labels = ["All Layers", "Layer 11"]
# --------- figure ---------
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), dpi=300)

# ===== subplot 1 =====
x1 = np.arange(len(alphas))
ax1 = axs[0]
ax1b = ax1.twinx()

bar1 = ax1b.bar(x1, tokens1, width=0.6, color="#4C78A8", label='Avg Reasoning Tokens')
line1 = ax1.plot(x1, acc1, color="#F58518", marker='o', linewidth=2, label='Accuracy')

ax1.set_zorder(3)
ax1.patch.set_visible(False)
ax1.set_xlabel('Adaptive alpha', fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=16)
ax1b.set_ylabel('Avg Reasoning Tokens', fontsize=16)

ax1.set_xticks(x1)
ax1.set_xticklabels([str(a) for a in alphas], fontsize=12)
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

lines = line1 + [bar1]
labels_ = [l.get_label() for l in line1] + [bar1.get_label()]
ax1.legend(lines, labels_, loc='upper right', fontsize=12)

# ===== subplot 2 =====
x2 = np.arange(len(labels))
w = 0.35
c_tok = "#2C5C8A"
c_acc = "#D55E00"

ax2 = axs[1]
ax2b = ax2.twinx()

bar_acc = ax2.bar(x2 - w/2, acc2, width=w, color="#F58518", label="Accuracy", zorder=1)
bar_tok = ax2b.bar(x2 + w/2, tokens2, width=w, color="#4C78A8", label="Avg Reasoning Tokens", zorder=1)

ax2.bar_label(bar_acc, fmt="%.3f", padding=4, fontsize=12)
ax2b.bar_label(bar_tok, fmt="%.1f", padding=4, fontsize=12)

ref1 = ax2.axhline(ref_acc, color=c_acc, linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Steer(10-14) Accuracy")
ref2 = ax2b.axhline(ref_tokens, color=c_tok, linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Steer(10-14) Tokens")

#ax2.set_xlabel("Setting", fontsize=16)
ax2.set_ylabel("Accuracy", fontsize=16)
ax2b.set_ylabel("Avg Reasoning Tokens", fontsize=16)

ax2.set_xticks(x2)
ax2.set_xticklabels(labels, fontsize=16)

ax2.set_ylim(0, max(max(acc2), ref_acc) * 1.5)
ax2b.set_ylim(0, 10000)

ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

handles = [bar_acc, bar_tok, ref1, ref2]
labels2 = ["Accuracy", "Avg Reasoning Tokens", "Steer(10-12) Accuracy @ alpha = 1.1", "Steer(10-12) Tokens @ alpha = 1.1"]
ax2.legend(handles, labels2, loc="upper left", bbox_to_anchor=(0.02, 1.02), framealpha=0.9, fontsize=12)

# keep ax2 on top of ax2b
ax2b.set_zorder(ax2.get_zorder() + 1)
ax2b.patch.set_visible(False)
ax2.patch.set_visible(False)

fig.tight_layout()
plt.show()


# %%
strengths = [0.00, -0.05, -0.10, -0.15, -0.20]
acc = [0.64, 0.64, 0.6333333333333333, 0.62, 0.6066666666666667]
tokens = [6459.206666666667, 5129.62, 4399.44, 3844.0733333333333, 3687.06]

ref_acc = 0.6466666666666666
ref_tokens = 4334.473333333333

x = np.arange(len(strengths))

fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)
ax2 = ax1.twinx()

bar = ax2.bar(x, tokens, width=0.6, color="#4C78A8", label="Avg Reasoning Tokens", zorder=1)
line = ax1.plot(x, acc, color="#F58518", marker="o", linewidth=2, label="Accuracy", zorder=3)

# ref lines
ax1.axhline(ref_acc, color="#D55E00", linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Ref Accuracy")
ax2.axhline(ref_tokens, color="#2C5C8A", linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Ref Tokens")

ax1.set_xlabel("Strength", fontsize=14)
ax1.set_ylabel("Accuracy", fontsize=14)
ax2.set_ylabel("Avg Reasoning Tokens", fontsize=14)

ax1.set_xticks(x)
ax1.set_xticklabels([str(s) for s in strengths])
ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

# legend
handles = line + [bar, ax1.lines[1], ax2.lines[0]]
labels = ["Accuracy", "Avg Reasoning Tokens", "Ref Accuracy", "Ref Tokens"]
ax1.legend(handles, labels, loc="upper right", fontsize=11)

ax1.set_zorder(2)
ax2.set_zorder(1)
ax1.patch.set_visible(False)
ax2.patch.set_visible(False)

fig.tight_layout()
plt.show()

# %%

# --------- subplot A ---------
labelsA = ["All Layers @ alpha = 0.3540", "Layer 12 @ alpha = 2.1243"]
accA = [0.6333333333333333, 0.6466666666666666]
tokensA = [4192.66, 5603.266666666666]

ref_acc = 0.6533333333333333
ref_tokens = 3938.66

# --------- subplot B ---------
strengthsB = [0.00, -0.05, -0.10, -0.15, -0.20]
accB = [0.64, 0.64, 0.6333333333333333, 0.62, 0.6066666666666667]
tokensB = [6459.206666666667, 5129.62, 4399.44, 3844.0733333333333, 3687.06]

# --------- figure ---------
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), dpi=300)

# ===== subplot A =====
xA = np.arange(len(labelsA))
w = 0.35
axA = axs[1]
axA2 = axA.twinx()

bar_acc = axA.bar(xA - w/2, accA, width=w, color="#4C78A8", label="Accuracy", zorder=1)
bar_tok = axA2.bar(xA + w/2, tokensA, width=w, color="#F58518", label="Avg Reasoning Tokens", zorder=1)

axA.bar_label(bar_acc, fmt="%.3f", padding=4, fontsize=12)
axA2.bar_label(bar_tok, fmt="%.1f", padding=4, fontsize=12)

ref1 = axA.axhline(ref_acc, color="#2C5C8A", linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Steer(10-14) Accuracy")
ref2 = axA2.axhline(ref_tokens, color="#D55E00", linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Steer(10-14) Tokens")

axA.set_ylabel("Accuracy", fontsize=16)
axA2.set_ylabel("Avg Reasoning Tokens", fontsize=16)
axA.set_xticks(xA)
axA.set_xticklabels(labelsA, fontsize=14)
axA.set_ylim(0, max(max(accA), ref_acc) * 1.45)
axA2.set_ylim(0, 8800)
axA.grid(True, axis="y", linestyle="--", alpha=0.3)

handlesA = [bar_acc, bar_tok, ref1, ref2]
labelsA_legend = ["Accuracy", "Avg Reasoning Tokens",
                  "Steer(10-14) Accuracy @ alpha = 0.95",
                  "Steer(10-14) Tokens @ alpha = 0.95"]
axA.legend(handlesA, labelsA_legend, loc="upper left",
           bbox_to_anchor=(0.02, 1.02), framealpha=0.9, fontsize=11)

# keep left axis on top
axA2.set_zorder(axA.get_zorder() + 1)
axA2.patch.set_visible(False)
axA.patch.set_visible(False)

# ===== subplot B =====
xB = np.arange(len(strengthsB))
axB = axs[0]
axB2 = axB.twinx()

barB = axB2.bar(xB, tokensB, width=0.6, color="#F58518", label="Avg Reasoning Tokens", zorder=1)
lineB = axB.plot(xB, accB, color="#4C78A8", marker="o", linewidth=2, label="Accuracy", zorder=3)

axB.axhline(ref_acc, color="#2C5C8A", linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Ref Accuracy")
axB2.axhline(ref_tokens, color="#D55E00", linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Ref Tokens")

axB.set_xlabel("Strength", fontsize=16)
axB.set_ylabel("Accuracy", fontsize=16)
axB2.set_ylabel("Avg Reasoning Tokens", fontsize=16)

axB.set_xticks(xB)
axB.set_xticklabels([str(s) for s in strengthsB], fontsize=12)
axB.grid(True, axis="y", linestyle="--", alpha=0.3)

handlesB = lineB + [barB, axB.lines[1], axB2.lines[0]]
labelsB = ["Accuracy", "Avg Reasoning Tokens", "Ref Accuracy", "Ref Tokens"]
axB.legend(handlesB, labelsB,
           loc="upper right",
           bbox_to_anchor=(1.0, 0.90),
           fontsize=11)

axB.set_zorder(2)
axB2.set_zorder(1)
axB.patch.set_visible(False)
axB2.patch.set_visible(False)

fig.tight_layout()
plt.show()

# %%
mean_vec_norm = [0, np.float32(1.775953), np.float32(2.9678245), np.float32(3.6077752), np.float32(5.45929), np.float32(6.006828), np.float32(6.8006973), np.float32(7.623228), np.float32(8.280823), np.float32(10.981863), np.float32(14.892502), np.float32(19.585176), np.float32(24.187458), np.float32(28.948988), np.float32(34.991203), np.float32(37.06481), np.float32(47.884583), np.float32(61.05658), np.float32(85.14428), np.float32(118.712326), np.float32(165.47403), np.float32(230.6632), np.float32(314.53043), np.float32(396.49765), np.float32(481.86218), np.float32(558.7757), np.float32(657.45465), np.float32(751.1294)]
cos_mean = [0, np.float32(0.88353133), np.float32(0.9072794), np.float32(0.9058813), np.float32(0.9059996), np.float32(0.9095177), np.float32(0.91088164), np.float32(0.9072138), np.float32(0.9126693), np.float32(0.91855186), np.float32(0.91880685), np.float32(0.92661095), np.float32(0.9239895), np.float32(0.921652), np.float32(0.9269673), np.float32(0.9188356), np.float32(0.9096881), np.float32(0.91242707), np.float32(0.8983774), np.float32(0.8873079), np.float32(0.8934437), np.float32(0.8812008), np.float32(0.8608046), np.float32(0.8429577), np.float32(0.82513696), np.float32(0.8056571), np.float32(0.7835336), np.float32(0.76506495)]
cos_std = [0, np.float32(0.07466827), np.float32(0.036372453), np.float32(0.036866277), np.float32(0.029055284), np.float32(0.03094469), np.float32(0.02950417), np.float32(0.026611954), np.float32(0.023672104), np.float32(0.022840623), np.float32(0.02327366), np.float32(0.021694206), np.float32(0.022910526), np.float32(0.02510912), np.float32(0.023656128), np.float32(0.026717585), np.float32(0.02819793), np.float32(0.027644249), np.float32(0.035019375), np.float32(0.038842868), np.float32(0.037341353), np.float32(0.043610416), np.float32(0.051714327), np.float32(0.060175974), np.float32(0.0676643), np.float32(0.07307237), np.float32(0.08045202), np.float32(0.08514235)]
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

# %%
fig, ax1 = plt.subplots(figsize=(6, 4), dpi=300)

# y1: Cosine Similarity Mean
ax1.plot(layers, cos_mean, color="#4C78A8", linestyle='--', label='Cosine Similarity Mean', linewidth=2.5)
ax1.set_ylabel('Cosine Similarity Mean', fontsize=12)
ax1.tick_params(axis='y')

# y2: Cosine Similarity Std
ax2 = ax1.twinx()
ax2.plot(layers, cos_std, color="#F58518", linestyle=':', label='Cosine Similarity Standard Deviation', linewidth=2.5)
ax2.set_ylabel('Cosine Similarity Standard Deviation', fontsize=12)
ax2.tick_params(axis='y')

ax1.set_xlabel('Layers')
ax1.set_title('Qwen3-VL-4B-Thinking Layer Analysis', fontsize=14)

lines = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


# %%
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), dpi=300)

# ===== subplot 1: layer analysis =====
ax1 = axs[0]
ax1.plot(layers, cos_mean, color="#4C78A8", linestyle="--",
         label="Cosine Similarity Mean", linewidth=2.5)
ax1.set_ylabel("Cosine Similarity Mean", fontsize=16)

ax1b = ax1.twinx()
ax1b.plot(layers, cos_std, color="#F58518", linestyle=":",
          label="Cosine Similarity Standard Deviation", linewidth=2.5)
ax1b.set_ylabel("Cosine Similarity Standard Deviation", fontsize=16)

ax1.set_xlabel("Layers", fontsize=16)
ax1.set_title("Qwen3-VL-4B-Thinking Layer Analysis", fontsize=16)

lines1 = ax1.get_lines() + ax1b.get_lines()
labels1 = [l.get_label() for l in lines1]
ax1.legend(lines1, labels1, loc="best", fontsize=11)

# ===== subplot 2: alpha vs acc/tokens =====
alphas = [0.0, 0.5, 0.8, 0.95, 1.1, 1.4]
acc = [0.64, 0.6533333333333333, 0.6266666666666667, 0.6533333333333333, 0.5933333333333334, 0.5866666666666667]
tokens = [6459.206666666667, 4925.113333333334, 4191.98, 3938.66, 3405.34, 3000.32]

x = np.arange(len(alphas))

ax2 = axs[1]
ax2b = ax2.twinx()

bar = ax2b.bar(x, tokens, width=0.6, color="#4C78A8", label="Avg Reasoning Tokens")
line = ax2.plot(x, acc, color="#F58518", marker="o", linewidth=2, label="Accuracy")

ax2.set_zorder(3)
ax2.patch.set_visible(False)
ax2.set_xlabel("Adaptive alpha", fontsize=16)
ax2.set_ylabel("Accuracy", fontsize=16)
ax2b.set_ylabel("Avg Reasoning Tokens", fontsize=16)

ax2.set_xticks(x)
ax2.set_xticklabels([str(a) for a in alphas])
ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

lines2 = line + [bar]
labels2 = [l.get_label() for l in line] + [bar.get_label()]
ax2.legend(lines2, labels2, loc="upper right", fontsize=11)

fig.tight_layout()
plt.show()

# %%

# ====== Plot 1: Layer analysis ======

# ====== Plot 2: alpha vs acc/tokens ======
alphas = [0.0, 0.8, 0.95, 1.1, 1.25, 1.4]
acc1 = [0.64, 0.64, 0.62, 0.6466666666666666, 0.6066666666666667, 0.6333333333333333]
tokens1 = [6459.206666666667, 4964.826666666667, 4619.74, 4405.326666666667, 4352.893333333333, 4185.3133333333335]

# ====== Plot 3: strength vs acc/tokens ======
strengths = [0.00, -0.05, -0.10, -0.15, -0.20]
acc_strength = [0.64, 0.64, 0.6333333333333333, 0.62, 0.6066666666666667]
tokens_strength = [6459.206666666667, 5129.62, 4399.44, 3844.0733333333333, 3687.06]

# ====== Plot 4: layer comparison (bar+ref) ======
labelsA = ["All Layers @ alpha = 0.4099", "Layer 12 @ alpha = 2.4597"]
accA = [0.6333333333333333, 0.6333333333333333]
tokensA = [6190.573333333334, 5603.266666666666]

ref_acc = 0.6466666666666666
ref_tokens = 4405.326666666667

# ====== figure ======
fig, axs = plt.subplots(2, 2, figsize=(14, 9), dpi=300)

# --- subplot (0,0): layer analysis ---
ax1 = axs[0, 0]
ax1.plot(layers, cos_mean, color="#4C78A8", linestyle="--",
         label="Cosine Similarity Mean", linewidth=2.5)
ax1.set_ylabel("Cosine Similarity Mean", fontsize=16)

ax1b = ax1.twinx()
ax1b.plot(layers, cos_std, color="#F58518", linestyle=":",
          label="Cosine Similarity Standard Deviation", linewidth=2.5)
ax1b.set_ylabel("Cosine Similarity Standard Deviation", fontsize=16)

ax1.set_xlabel("Layers", fontsize=16)
#ax1.set_title("Qwen3-VL-4B-Thinking Layer Analysis", fontsize=16)

lines1 = ax1.get_lines() + ax1b.get_lines()
labels1 = [l.get_label() for l in lines1]
ax1.legend(lines1, labels1, loc="best", fontsize=11)

# --- subplot (0,1): alpha vs acc/tokens ---
ax2 = axs[0, 1]
ax2b = ax2.twinx()

x2 = np.arange(len(alphas))
bar2 = ax2b.bar(x2, tokens1, width=0.6, color="#4C78A8", label="Avg Reasoning Tokens")
line2 = ax2.plot(x2, acc1, color="#F58518", marker="o", linewidth=2, label="Accuracy")

ax2.set_zorder(3)
ax2.patch.set_visible(False)
ax2.set_xlabel("Adaptive alpha", fontsize=16)
ax2.set_ylabel("Accuracy", fontsize=16)
ax2b.set_ylabel("Avg Reasoning Tokens", fontsize=16)

ax2.set_xticks(x2)
ax2.set_xticklabels([str(a) for a in alphas], fontsize=12)
ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

lines2 = line2 + [bar2]
labels2 = [l.get_label() for l in line2] + [bar2.get_label()]
ax2.legend(lines2, labels2, loc="upper right", fontsize=11)

# --- subplot (1,0): strength vs acc/tokens ---
ax3 = axs[1, 0]
ax3b = ax3.twinx()

x3 = np.arange(len(strengths))
bar3 = ax3b.bar(x3, tokens_strength, width=0.6, color="#4C78A8", label="Avg Reasoning Tokens", zorder=1)
line3 = ax3.plot(x3, acc_strength, color="#F58518", marker="o", linewidth=2, label="Accuracy", zorder=3)

ax3.axhline(ref_acc, color="#D55E00", linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Ref Accuracy")
ax3b.axhline(ref_tokens, color="#2C5C8A", linestyle="--", linewidth=2, alpha=0.9, zorder=10, label="Ref Tokens")

ax3.set_xlabel("Strength", fontsize=16)
ax3.set_ylabel("Accuracy", fontsize=16)
ax3b.set_ylabel("Avg Reasoning Tokens", fontsize=16)

ax3.set_xticks(x3)
ax3.set_xticklabels([str(s) for s in strengths], fontsize=12)
ax3.grid(True, axis="y", linestyle="--", alpha=0.3)

handles3 = line3 + [bar3, ax3.lines[1], ax3b.lines[0]]
labels3 = ["Accuracy", "Avg Reasoning Tokens", "Steer(10-14) Accuracy @ alpha = 1.1", "Steer(10-14) Tokens @ alpha = 1.1"]
ax3.legend(handles3, labels3,
           loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize=11)

ax3.set_zorder(2)
ax3b.set_zorder(1)
ax3.patch.set_visible(False)
ax3b.patch.set_visible(False)

# --- subplot (1,1): layer comparison (bar+ref) ---
ax4 = axs[1, 1]
ax4b = ax4.twinx()

x4 = np.arange(len(labelsA))
w = 0.35
bar_acc = ax4.bar(x4 - w/2, accA, width=w, color="#F58518", label="Accuracy", zorder=1)
bar_tok = ax4b.bar(x4 + w/2, tokensA, width=w, color="#4C78A8", label="Avg Reasoning Tokens", zorder=1)

ax4.bar_label(bar_acc, fmt="%.3f", padding=4, fontsize=12)
ax4b.bar_label(bar_tok, fmt="%.1f", padding=4, fontsize=12)

ref1 = ax4.axhline(ref_acc, color="#D55E00", linestyle="--", linewidth=2, alpha=0.9, zorder=10,
                   label="Steer(10-14) Accuracy")
ref2 = ax4b.axhline(ref_tokens, color="#2C5C8A", linestyle="--", linewidth=2, alpha=0.9, zorder=10,
                    label="Steer(10-14) Tokens")

ax4.set_ylabel("Accuracy", fontsize=16)
ax4b.set_ylabel("Avg Reasoning Tokens", fontsize=16)
ax4.set_xticks(x4)
ax4.set_xticklabels(labelsA, fontsize=16)
ax4.set_ylim(0, max(max(accA), ref_acc) * 1.45)
ax4b.set_ylim(3500, 7500)
ax4.grid(True, axis="y", linestyle="--", alpha=0.3)

handles4 = [bar_acc, bar_tok, ref1, ref2]
labels4 = ["Accuracy", "Avg Reasoning Tokens",
           "Steer(10-14) Accuracy @ alpha = 1.1",
           "Steer(10-14) Tokens @ alpha = 1.1"]
ax4.legend(handles4, labels4, loc="upper left",
           bbox_to_anchor=(0.02, 1.02), framealpha=0.9, fontsize=11)

ax4b.set_zorder(ax4.get_zorder() + 1)
ax4b.patch.set_visible(False)
ax4.patch.set_visible(False)

fig.tight_layout()
plt.show()



