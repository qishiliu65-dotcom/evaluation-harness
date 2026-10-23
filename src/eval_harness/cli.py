import argparse
# argparse 是 Python 自带的标准库模块，专门处理命令行参数。
from .dataset import load_evaluation_cases
from .systems import LLMSystem
from .runner import run_batch_evaluation


def main():
    # ArgumentParser是在创建一个命令行参数解析器对象，description是描述这个 CLI 是干什么的，是给帮助页面看的
    parser = argparse.ArgumentParser(description="Run evaluation cases against an LLM system.")

    # 添加 --dataset 参数，required=True意思是这个参数必须提供；help也是给帮助页面看的
    parser.add_argument("--dataset", required=True, help="Path to the JSONL evaluation dataset.")

    # 添加 --model 参数
    parser.add_argument("--model", required=True, help="Model name to evaluate.")

    #parse_args()是解析命令行参数
    args = parser.parse_args()

    dataset_path = args.dataset
    model = args.model

    cases = load_evaluation_cases(dataset_path)
    system = LLMSystem(model)
    result = run_batch_evaluation(cases, system)

    print(result)


if __name__ == "__main__":
    main()