import re
import pandas as pd
import matplotlib.pyplot as plt

# === Step 1: 读取 textcat 输出 ===
results = []
with open("langid_results.txt") as f:
    for line in f:
        line = line.strip()
        if not line or not line.startswith(("en.model", "sp.model")):
            continue

        model, filename = line.split(maxsplit=1)

        # === Step 2: 提取 gold & pred ===
        gold = "english" if "en" in filename else "spanish"
        pred = "english" if "en.model" in model else "spanish"

        # === Step 3: 提取文件长度 ===
        # 适配文件名 like sp.500.09 / en.80.02
        m = re.search(r"\.(\d+)\.", filename)
        if not m:
            continue
        length = int(m.group(1))

        correct = int(pred == gold)
        results.append((gold, length, correct))

# === Step 4: 计算错误率 ===
df = pd.DataFrame(results, columns=["language", "length", "correct"])
df["error"] = 1 - df["correct"]

# 按语言+长度分组求平均错误率
grouped = df.groupby(["language", "length"])["error"].mean().reset_index()

# === Step 5: 画图（两条折线） ===
plt.figure(figsize=(7,4))

for lang, subdf in grouped.groupby("language"):
    plt.plot(subdf["length"], subdf["error"], marker="o", linestyle="-", label=lang.capitalize())

plt.xlabel("File length (characters)")
plt.ylabel("Average error rate")
plt.title("Language ID Performance vs File Length (dev set)")
plt.grid(True)
plt.legend(title="True language")
plt.tight_layout()
plt.savefig("langid_performance_vs_length_split.png", dpi=150)
plt.show()
