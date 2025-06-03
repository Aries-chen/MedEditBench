import argparse
import json
import numpy as np


def compute_rates(results):
    both_entries = [r for r in results if ("nq_success" in r and "loc_success" in r)]
    
    ori_count   = len(both_entries)
    ori_success = sum(1 for r in both_entries if r.get("ori_q_success"))
    nq_success  = sum(1 for r in both_entries if r["nq_success"])
    loc_success = sum(1 for r in both_entries if r["loc_success"])
    
    ori_rate = ori_success / ori_count if ori_count else 0.0
    nq_rate  = nq_success  / ori_count if ori_count else 0.0
    loc_rate = loc_success / ori_count if ori_count else 0.0
    
    avg_rate = np.mean([ori_rate, nq_rate, loc_rate]) if ori_count else 0.0
    
    return ori_rate, nq_rate, loc_rate, avg_rate


def compute_semantic(results):
    both_entries = [r for r in results if ("nq_success" in r and "loc_success" in r)]
    
    rougeLs = [r["rougeL_score"] for r in both_entries if "rougeL_score" in r]
    bleus   = [r["bleu_score"]  for r in both_entries if "bleu_score"  in r]
    
    rouge_mean = np.mean(rougeLs) if rougeLs else None
    bleu_mean  = np.mean(bleus)   if bleus   else None
    

    semantic_mean = np.mean([rouge_mean, bleu_mean])
    
    return rouge_mean, bleu_mean, semantic_mean


def main():
    parser = argparse.ArgumentParser(
        description="Offline compute edit success rates and semantic scores"
    )
    parser.add_argument("--input_json",      "-i", type=str, required=True,
                        help="eidted JSON path")
    
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        results = json.load(f)

    ori_r, nq_r, loc_r, avg_r = compute_rates(results)
    rouge_m, bleu_m, sem_m    = compute_semantic(results)

    print(f"ori_success_rate:   {ori_r:.3f}")
    print(f"nq_success_rate:    {nq_r:.3f}")
    print(f"loc_success_rate:   {loc_r:.3f}")
    print(f"avg_success_rate:   {avg_r:.3f}")
    if rouge_m is not None:
        print(f"rougeL_score_mean:  {rouge_m:.3f}")
        print(f"bleu_score_mean:    {bleu_m:.3f}")
        print(f"semantic_sim_mean:  {sem_m:.3f}")


if __name__ == "__main__":
    main()

# Example usage:
# python compute_edit_metrics.py --input_json xxx/xxx.json
