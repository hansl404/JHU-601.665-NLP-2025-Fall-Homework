import re
import sys
from pathlib import Path

# === 用法示例 ===
# python analyze_cross_entropy.py logs/find_best_d_ensp_results.log

def parse_log_file(log_path: str):
    log_text = Path(log_path).read_text(encoding="utf-8")

    # 匹配 cross-entropy 和 token 数
    pattern = re.compile(
        r"Overall cross-entropy:\s*([0-9.]+)\s*bits per token, total tokens:\s*([0-9]+)",
        re.IGNORECASE
    )

    matches = pattern.findall(log_text)
    # 同时提取对应模型名
    run_pattern = re.compile(r"\[Running\]\s*python\s+fileprob\.py\s+(\S+)\s", re.IGNORECASE)
    run_cmds = run_pattern.findall(log_text)

    results = []
    for i, m in enumerate(matches):
        try:
            model_name = Path(run_cmds[i]).stem
        except IndexError:
            model_name = f"model_{i}"
        ce, tokens = float(m[0]), int(m[1])
        bits = ce * tokens
        results.append((model_name, ce, tokens, bits))
    return results


def summarize(results):
    """
    自动按语言对合并，比如 en/sp 或 gen/spam。
    """
    grouped = {}
    for model_name, ce, tokens, bits in results:
        # 推断语言 (前缀)
        lang = model_name.split("_")[0]
        grouped.setdefault(lang, []).append((model_name, ce, tokens, bits))

    # 按 (en, sp) 或 (gen, spam) 合并
    lang_keys = list(grouped.keys())
    if len(lang_keys) < 2:
        print("⚠️ 只找到一个语言的结果，无法平均。")
        return

    lang1, lang2 = lang_keys[:2]
    combined = []
    for i in range(min(len(grouped[lang1]), len(grouped[lang2]))):
        m1 = grouped[lang1][i]
        m2 = grouped[lang2][i]

        # 从模型名提取 C 或 d 标签
        tag = ""
        for token in ["C", "d"]:
            if f"_{token}" in m1[0]:
                tag = m1[0].split(f"_{token}", 1)[1]
                break

        total_bits = m1[3] + m2[3]
        total_tokens = m1[2] + m2[2]
        avg_ce = total_bits / total_tokens
        combined.append((tag, avg_ce))

    print(f"\n==== Averaged Cross-Entropy Summary for {Path(log_path).name} ====")
    print(f"{'Model tag':<25}{'Average CE (bits/token)':>30}")
    print("-" * 55)
    for tag, avg_ce in combined:
        print(f"{tag:<25}{avg_ce:>30.4f}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_cross_entropy.py <log_path>")
        sys.exit(1)

    log_path = sys.argv[1]
    results = parse_log_file(log_path)
    summarize(results)
