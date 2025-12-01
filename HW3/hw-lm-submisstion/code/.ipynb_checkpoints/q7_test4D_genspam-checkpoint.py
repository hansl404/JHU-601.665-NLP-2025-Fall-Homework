import os

best_C = 0.5          
dims = [10, 50, 200]
dev_base = "../data/gen_spam/dev"
domains = ["gen","spam"]
log_path = "logs/find_best_d_genspam_results.log"

os.makedirs("logs", exist_ok=True)

with open(log_path, "w", encoding="utf-8") as f_log:
    f_log.write("==== Cross-Entropy Evaluation for Different d ====\n\n")

    for d in dims:
        for domain in domains:
            model_path = f"{domain}_C{best_C}_d{d}.model"
            dev_file = f"{dev_base}/{domain}/*"
            cmd = f"python fileprob.py {model_path} {dev_file}"
            f_log.write(f"[Running] {cmd}\n")
            print(f"Running: {cmd}")

            stream = os.popen(cmd)
            output = stream.read()
            stream.close()

            f_log.write(output)
            f_log.write("\n" + "="*60 + "\n")

