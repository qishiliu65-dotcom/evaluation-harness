from .models import HumanLabel



def compute_raw_agreement(labels_a: list[HumanLabel], labels_b: list[HumanLabel]) -> float:
    """按 case_id 对齐两组标签，计算原始一致率。

    计算公式：
        一致率 = 标签相同的样本数 / 配对样本总数

    返回：
        0.0～1.0 之间的浮点数，不进行四舍五入。

    抛出 ValueError 的情况：
        1. 任意一组标签为空。
        2. 任意一组存在重复的 case_id。
        3. 两组包含的 case_id 不完全相同。
    """

    if not labels_a or not labels_b:
        raise ValueError("Label lists must not be empty")

    # 2. 将 A 组转成字典：case_id → label。
    #
    # 例如：
    # [
    #     HumanLabel(case_id="001", label="PASS"),
    #     HumanLabel(case_id="002", label="FAIL"),
    # ]
    #
    # 转换后：
    # {"001": "PASS", "002": "FAIL"}
    a_by_id: dict[str, str] = {}

    for item in labels_a:
        # 字典的 in 运算检查 key 是否存在。
        #
        # 如果当前 ID 已经存在，说明 A 组重复标注了同一个样本。
        # 必须在赋值之前检查，否则新值会覆盖旧值。
        if item.case_id in a_by_id:
            raise ValueError(
                f"Duplicate case_id in labels_a: {item.case_id}"
            )

        # 将当前样本 ID 作为 key，标签作为 value。
        a_by_id[item.case_id] = item.label

    # 3. 用相同方式转换 B 组，并检查重复 ID。
    b_by_id: dict[str, str] = {}

    for item in labels_b:
        if item.case_id in b_by_id:
            raise ValueError(
                f"Duplicate case_id in labels_b: {item.case_id}"
            )

        b_by_id[item.case_id] = item.label

    # 4. 检查两组是否包含完全相同的样本。
    #
    # set(字典) 得到字典所有 key 组成的集合。
    # 集合比较不受顺序影响：
    # {"001", "002"} 与 {"002", "001"} 相等。
    #
    # 这里比较 ID，而不是比较标签。
    # 标签不同属于正常分歧，后面需要统计。
    if set(a_by_id) != set(b_by_id):
        raise ValueError(
            "Label lists must contain the same case_ids"
        )

    # 5. 统计标签一致的样本数量。
    agreement_count = 0

    # 遍历字典时，默认依次取出 key，即 case_id。
    for case_id in a_by_id:
        # 通过同一个 ID，分别读取 A、B 对这道题的标签。
        #
        # 前面已经确认两组 ID 相同，所以 B 中一定存在该 ID。
        if a_by_id[case_id] == b_by_id[case_id]:
            agreement_count += 1

    # 6. 计算并返回一致率。
    #
    # len(a_by_id) 是配对样本总数。
    # 前面的检查保证它大于 0，且与 B 的样本总数相同。
    #
    # Python 的 / 返回浮点数：
    # 2 / 3 → 0.6666666666666666
    # 3 / 3 → 1.0
    # 0 / 3 → 0.0
    return agreement_count / len(a_by_id)


def compute_cohens_kappa(labels_a: list[HumanLabel], labels_b: list[HumanLabel]) -> float:
    if not labels_a or not labels_b:
        raise ValueError("Label lists must not be empty")

    # 实际有多少样本标签一致
    po = compute_raw_agreement(labels_a, labels_b)

    a_pass_count = 0
    b_pass_count = 0

    for item in labels_a:
        if item.label == "PASS":
            a_pass_count += 1

    for item in labels_b:
        if item.label == "PASS":
            b_pass_count += 1

    total = len(labels_a)
    a_pass_ratio = a_pass_count / total
    b_pass_ratio = b_pass_count / total

    a_fail_ratio = 1 - a_pass_ratio
    b_fail_ratio = 1 - b_pass_ratio

    # 按各自比例随机打标签，预计会有多少碰巧一致
    pe = (a_pass_ratio * b_pass_ratio + a_fail_ratio * b_fail_ratio)

    if pe == 1.0:
        raise ValueError(
        "Cohen's kappa is undefined when expected agreement is 1"
        )

    # 返回数据表示：扣除偶然一致的影响后，两方的判断还有多大程度的一致性
    return (po - pe) / (1 - pe)





# 混淆矩阵：统计 Judge 判对、错放和错杀的数量。
# 以 PASS 为正类，第一个参数是参考标签（Gold/Human），第二个是 Judge 标签。
# TP：参考 PASS，Judge PASS → 正确放行。
# TN：参考 FAIL，Judge FAIL → 正确拒绝。
# FP：参考 FAIL，Judge PASS → 错放。
# FN：参考 PASS，Judge FAIL → 错杀。
# 若参考为 Human，上述对错仅表示相对于人工标签的判断。
# 按 case_id 对齐；拒绝空输入、重复 ID 和两边 ID 不一致。
# 返回 {"tp": 数量, "tn": 数量, "fp": 数量, "fn": 数量}。

def compute_confusion_matrix(reference_labels: list[HumanLabel], predicted_labels: list[HumanLabel],) -> dict[str, int]:
    """按 case_id 对齐标签，以 PASS 为正类统计混淆矩阵。"""

    # 1. 拒绝空输入。
    if not reference_labels or not predicted_labels:
        raise ValueError("Label lists must not be empty")

    # 2. 将参考标签转换为 case_id → label 字典。
    reference_by_id: dict[str, str] = {}

    for item in reference_labels:
        if item.case_id in reference_by_id:
            raise ValueError(
                f"Duplicate case_id in reference_labels: {item.case_id}"
            )

        reference_by_id[item.case_id] = item.label

    # 3. 将预测标签转换为 case_id → label 字典。
    predicted_by_id: dict[str, str] = {}

    for item in predicted_labels:
        if item.case_id in predicted_by_id:
            raise ValueError(
                f"Duplicate case_id in predicted_labels: {item.case_id}"
            )

        predicted_by_id[item.case_id] = item.label

    # 4. 确保两边包含完全相同的样本。
    if set(reference_by_id) != set(predicted_by_id):
        raise ValueError(
            "Label lists must contain the same case_ids"
        )

    # 5. 初始化四种组合的计数。
    matrix = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }

    # 6. 按同一个 case_id 取出两边的标签并分类。
    for case_id in reference_by_id:
        reference = reference_by_id[case_id]
        predicted = predicted_by_id[case_id]

        if reference == "PASS" and predicted == "PASS":
            matrix["tp"] += 1

        elif reference == "FAIL" and predicted == "FAIL":
            matrix["tn"] += 1

        elif reference == "FAIL" and predicted == "PASS":
            matrix["fp"] += 1

        elif reference == "PASS" and predicted == "FAIL":
            matrix["fn"] += 1

    # 7. 全部样本处理完成后，返回计数字典。
    return matrix