import os

# === 固定参数 ===
vocab = "vocab-ensp.txt"
epochs = 10
device = "cuda"
best_C = 0.5         
dims = [50, 200]   
langs = ["en", "sp"]   

for d in dims:
    lexicon = f"../lexicons/words-gs-only-{d}.txt"
    for lang in langs:
        output = f"{lang}_C{best_C}_d{d}.model"
        train_path = f"../data/english_spanish/train/{lang}.1K"
        log_file = f"logs/{lang}_C{best_C}_d{d}.out"

        os.makedirs("logs", exist_ok=True)

        cmd = (
            f"nohup python -u train_lm.py {vocab} log_linear {train_path} "
            f"--output {output} "
            f"--lexicon {lexicon} "
            f"--l2_regularization {best_C} "
            f"--epochs {epochs} "
            f"--device {device} "
            f"> {log_file} 2>&1 &"
        )

        print(f"Running: {cmd}")
        os.system(cmd)
