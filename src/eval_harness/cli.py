import argparse
# argparse 是 Python 自带的标准库模块，专门处理命令行参数。
from .dataset import load_evaluation_cases
from .systems import LLMSystem
from .runner import run_batch_evaluation
from .evaluator import (
    evaluate_exact_match,
    evaluate_normalized_match,
)
from .reproducibility import compute_file_sha256
from .reporting import save_evaluation_result_json


def main():
    # ArgumentParser是在创建一个命令行参数解析器对象，description是描述这个 CLI 是干什么的，是给帮助页面看的
    parser = argparse.ArgumentParser(description="Run evaluation cases against an LLM system.")

    # 添加 --dataset 参数，required=True意思是这个参数必须提供；help也是给帮助页面看的
    parser.add_argument("--dataset", required=True, help="Path to the JSONL evaluation dataset.")

    # 添加 --model 参数
    parser.add_argument("--model", required=True, help="Model name to evaluate.")

    parser.add_argument("--evaluator", choices=["exact", "normalized"], default="exact", help="Evaluation strategy to use.")

    parser.add_argument("--output", default="results/eval_result.json", help="Path to save the evaluation result JSON."
)
    
    #parse_args()是解析命令行参数
    args = parser.parse_args()
    output_path = args.output

    evaluators = {
        "exact": evaluate_exact_match,
        "normalized": evaluate_normalized_match,
    }
    evaluator = evaluators[args.evaluator]


    dataset_path = args.dataset
    model = args.model
    dataset_sha256 = compute_file_sha256(dataset_path)
    
    cases = load_evaluation_cases(dataset_path)
    system = LLMSystem(model)
    result = run_batch_evaluation(cases, system, evaluator)
    result.dataset_version = "v0.1"
    result.dataset_sha256 = dataset_sha256

    save_evaluation_result_json(result, output_path)
    
    print(result)


if __name__ == "__main__":
    main()