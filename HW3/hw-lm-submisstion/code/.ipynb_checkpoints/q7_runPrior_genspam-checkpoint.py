import os

# ==== 固定参数 ====
C = 0.5
d = 200
priors = [0.5, 0.6, 0.7,0.8, 0.9]   # p(gen) 取值
device = "cuda"

# 模型路径
gen_model = f"gen_C{C}_d{d}.model"
spam_model = f"spam_C{C}_d{d}.model"

# 数据路径 
base_path = "../data/gen_spam/test"
test_sets = {
    "all": f"{base_path}/*/*",
    "gen_only": f"{base_path}/gen/*",
    "spam_only": f"{base_path}/spam/*",
}

# 输出目录
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==== 批量运行 textcat.py ====
for prior in priors:
    for name, path in test_sets.items():
        log_file = f"logs/textcat_C{C}_d{d}_p{prior}_{name}.out"
        result_file = f"results/textcat_C{C}_d{d}_p{prior}_{name}.txt"

        cmd = (
            f"nohup python -u textcat.py {gen_model} {spam_model} {prior} {path} --device cuda"
            f"> {log_file} 2>&1 &"
        )

        print(f"[Launching] Prior={prior} | Test={name}")
        os.system(cmd)
