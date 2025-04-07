import os
import requests

if __name__ == "__main__":
    try:
        # 这里尝试连接hugging face连接不上就换国内镜像源
        response = requests.get("https://huggingface.co", timeout=5)
        if response.status_code == 200:
            print("成功连接到 Hugging Face")
        else:
            print(f"连接失败，状态码: {response.status_code}")
    except requests.exceptions.RequestException:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"无法连接到 Hugging Face:换源到{os.environ['HF_ENDPOINT']}")

from datasets import get_dataset_config_names, get_dataset_split_names
import json


def fetch_all_conditions_from_huggingface(dataset_name):
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
    # print(json.dumps(conditions, indent=2))
    # 返回conditions的key用数组存储
    print(conditions[0][0])
