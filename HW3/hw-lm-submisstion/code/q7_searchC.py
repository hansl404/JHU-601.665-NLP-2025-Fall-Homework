import os

# 固定参数
lexicon = "../lexicons/words-gs-only-10.txt"
vocab = "genspam-vocab.txt"
epochs = 10
device = "cuda"
Cs = [0, 0.1, 0.5, 1, 5]

for C in Cs:
    for domain in ["gen", "spam"]:
        output = f"{domain}_C{C}_d10.model"
        train_path = f"../data/gen_spam/train/{domain}"
        log_file = f"logs/{domain}_C{C}_d10.out"

        os.makedirs("logs", exist_ok=True)

        cmd = (
            f"nohup python -u train_lm.py {vocab} log_linear {train_path} "
            f"--output {output} "
            f"--lexicon {lexicon} "
            f"--l2_regularization {C} "
            f"--epochs {epochs} "
            f"--device {device} "
            f"> {log_file} 2>&1 &"
        )

        print(f"Running: {cmd}")
        os.system(cmd)
