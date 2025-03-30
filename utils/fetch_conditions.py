from datasets import get_dataset_config_names, get_dataset_split_names
import json


def fetch_all_conditions_from_huggingface(dataset_name):
    # TODO 换源
    """所有数据集的subset和split
    具体见网页https://huggingface.co/datasets/BFDS-Project/Bearing-Fault-Diagnosis-System
    Args:
        dataset_name (str): 数据集名称
    Returns:
        dict: 包含配置名称和对应分割信息的字典
    """
    # 获取数据集的可用配置
    available_configs = get_dataset_config_names(dataset_name)

    # 动态生成 conditions
    conditions = {}
    for config in available_configs:
        splits = get_dataset_split_names(dataset_name, config)
        conditions[config] = list(splits)  # 将分割信息存储为列表
    return conditions


if __name__ == "__main__":
    dataset_name = "BFDS-Project/Bearing-Fault-Diagnosis-System"
    conditions = fetch_all_conditions_from_huggingface(dataset_name)
    print("huggingface上的数据集配置和分割信息:")
    print(json.dumps(conditions, indent=2))
