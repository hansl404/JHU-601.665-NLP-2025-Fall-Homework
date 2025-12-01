import os

best_C = 0.5            
dims = [10, 50, 200]
langs = ["english", "spanish"]
dev_base = "../data/english_spanish/dev"
log_path = "logs/find_best_d_ensp_results.log"

os.makedirs("logs", exist_ok=True)

with open(log_path, "w", encoding="utf-8") as f_log:
    f_log.write("==== Cross-Entropy Evaluation for English/Spanish ====\n\n")

    for d in dims:
        for lang in langs:
            if lang == "english": prefix = "en"
            if lang == "spanish": prefix = "sp"
            model_path = f"{prefix}_C{best_C}_d{d}.model"
            dev_file = f"{dev_base}/{lang}/*/*"
            cmd = f"python fileprob.py {model_path} {dev_file} --device cuda"
            f_log.write(f"[Running] {cmd}\n")
            print(f"Running: {cmd}")

            stream = os.popen(cmd)
            output = stream.read()
            stream.close()

            f_log.write(output)
            f_log.write("\n" + "="*60 + "\n")


