import os
import re
import pandas as pd

# 文件所在目录（修改成你的路径）
result_dir = "./logs/"  

# 匹配文件名格式：textcat_C{C}_d{d}_p{p}_*.out
pattern = re.compile(r"textcat_C([\d.]+)_d(\d+)_p([\d.]+)")

rows = []

for filename in os.listdir(result_dir):
    if not filename.startswith("textcat_C") or not filename.endswith(".out"):
        continue
    
    match = pattern.search(filename)
    if not match:
        continue

    C_val, d_val, p_val = match.groups()
    file_path = os.path.join(result_dir, filename)
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if len(lines) < 2:
            continue
        last_two = lines[-2:]
    
    # 解析“xx files were more probably from gen/spam_C... (xx.xx%)”
    total_match = re.findall(r"(\d+)\s+files.*\(([\d.]+)%\)", " ".join(last_two))
    if len(total_match) == 2:
        gen_files, gen_percent = total_match[0]
        spam_files, spam_percent = total_match[1]
        gen_files, spam_files = int(gen_files), int(spam_files)
        gen_percent, spam_percent = float(gen_percent), float(spam_percent)
        total = gen_files + spam_files
    else:
        continue

    # 假设你的 dev 数据集中真实的 gen:spam 比例为 2:1，可根据任务补充 error 定义
    # 这里仅记录 raw 分类比例
    rows.append({
        "C": C_val,
        "d": d_val,
        "p(gen)": p_val,
        "#gen_pred": gen_files,
        "%gen_pred": gen_percent,
        "#spam_pred": spam_files,
        "%spam_pred": spam_percent,
        "Total": total,
    })

# 构建表格
df = pd.DataFrame(rows)
df = df.sort_values(by=["C", "d", "p(gen)"])

# 打印表格
print(df.to_string(index=False))

# 保存为 CSV 方便后续导入 LaTeX 或 Excel
df.to_csv("textcat_summary.csv", index=False)
print("\n✅ Saved summary to textcat_summary.csv")
