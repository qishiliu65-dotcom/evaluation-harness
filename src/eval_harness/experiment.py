from .agreement import (
    compute_raw_agreement,
    compute_cohens_kappa,
    compute_confusion_matrix,
)

from .models import (
    HumanLabel,
    JudgeExperimentMetrics,
    JudgeExperimentSummary,
)


def build_experiment_summary(
    human_labels: list[HumanLabel],
    gold_labels: list[HumanLabel],
    strict_labels: list[HumanLabel],
    lenient_labels: list[HumanLabel],
    dataset_version: str,
    model: str,
) -> JudgeExperimentSummary:
    """计算 Strict / Lenient Judge 的完整实验指标并返回结构化结果。"""

    # 1. 计算 Strict Judge 的指标。
    strict_metrics = JudgeExperimentMetrics(
        # Strict Judge 与 Human 的原始一致率。
        raw_agreement_with_human=compute_raw_agreement(
            strict_labels,
            human_labels,
        ),

        # Strict Judge 与 Human 的 Cohen's kappa。
        kappa_with_human=compute_cohens_kappa(
            strict_labels,
            human_labels,
        ),

        # Strict Judge 与 Gold 的原始一致率。
        #
        # 这里我们把它解释成：
        # Judge 相对于 Gold reference 的 correctness。
        raw_agreement_with_gold=compute_raw_agreement(
            strict_labels,
            gold_labels,
        ),

        # Strict Judge 与 Gold 的 Cohen's kappa。
        kappa_with_gold=compute_cohens_kappa(
            strict_labels,
            gold_labels,
        ),

        # Human 是 reference，
        # Strict Judge 是 predicted。
        confusion_vs_human=compute_confusion_matrix(
            human_labels,
            strict_labels,
        ),

        # Gold 是 reference，
        # Strict Judge 是 predicted。
        confusion_vs_gold=compute_confusion_matrix(
            gold_labels,
            strict_labels,
        ),
    )

    # 2. 计算 Lenient Judge 的指标。
    lenient_metrics = JudgeExperimentMetrics(
        raw_agreement_with_human=compute_raw_agreement(
            lenient_labels,
            human_labels,
        ),

        kappa_with_human=compute_cohens_kappa(
            lenient_labels,
            human_labels,
        ),

        raw_agreement_with_gold=compute_raw_agreement(
            lenient_labels,
            gold_labels,
        ),

        kappa_with_gold=compute_cohens_kappa(
            lenient_labels,
            gold_labels,
        ),

        confusion_vs_human=compute_confusion_matrix(
            human_labels,
            lenient_labels,
        ),

        confusion_vs_gold=compute_confusion_matrix(
            gold_labels,
            lenient_labels,
        ),
    )

    # 3. 将两组指标和实验元数据组合成完整 summary。
    return JudgeExperimentSummary(
        dataset_version=dataset_version,
        model=model,
        strict=strict_metrics,
        lenient=lenient_metrics,
    )