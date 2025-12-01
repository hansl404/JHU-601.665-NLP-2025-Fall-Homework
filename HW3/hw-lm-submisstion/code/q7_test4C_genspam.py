import os

# === 基本配置 ===
vocab = "genspam-vocab.txt"
device = "cuda"
Cs = [0, 0.1, 0.5, 1, 5]
domains = ["gen","spam"]
dev_base = "../data/gen_spam/dev"
log_path = "logs/find_best_c_genspam_results.log"

os.makedirs("logs", exist_ok=True)

with open(log_path, "w", encoding="utf-8") as f_log:
    f_log.write("==== Gen/Spam Cross-Entropy Evaluation ====\n\n")

    for C in Cs:
        for domain in domains:
            model_path = f"{domain}_C{C}_d10.model"
            dev_file = f"{dev_base}/{domain}/*"

            cmd = f"python fileprob.py {model_path} {dev_file}"

            f_log.write(f"\n[Running] {cmd}\n")
            print(f"Running: {cmd}")

            stream = os.popen(cmd)
            output = stream.read()
            stream.close()

            f_log.write(output)
            f_log.write("\n" + "=" * 60 + "\n")

    f_log.write("\nAll evaluations finished.\n")

print(f"\n✅ done. Results saved in {log_path}")