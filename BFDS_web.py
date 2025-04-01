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


import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
from BFDS_train import Argument, update_args_param
import pandas as pd
import torch
from utils.predict import predict

import logging
import warnings
from datetime import datetime


from utils.logger import setlogger
from utils.train import train_utils
from utils.fetch_conditions import fetch_all_conditions_from_huggingface

dataset_name = "BFDS-Project/Bearing-Fault-Diagnosis-System"
conditions = fetch_all_conditions_from_huggingface(dataset_name)

# 设置 Matplotlib 的后端为非交互式后端
matplotlib.use("Agg")
plt.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "font.size": 14,
        "font.serif": "STIXGeneral",
        "font.family": ["Arial", "Microsoft YaHei"],
        "axes.unicode_minus": False,
    }
)

# 初始化 Argument 实例
args = Argument()


# 更新参数的函数
def transfer_learning(
    source_config,
    source_split,
    target_config,
    target_split,
    normalize_type,
    model_name,
    bottleneck,
    bottleneck_num,
    batch_size,
    cuda_device,
    max_epoch,
    num_workers,
    opt,
    momentum,
    weight_decay,
    lr,
    lr_scheduler,
    gamma,
    steps_start,
    steps_end,
    middle_epoch,
    distance_option,
    distance_loss,
    distance_tradeoff,
    distance_lambda,
    adversarial_option,
    adversarial_loss,
    hidden_size,
    grl_option,
    grl_lambda,
    adversarial_tradeoff,
    adversarial_lambda,
    wavelet,
):
    args_params_dict = {
        "transfer_task": [[source_config, source_split], [target_config, target_split]],
        "normalize_type": normalize_type,
        "model_name": model_name,
        "bottleneck": bottleneck,
        "bottleneck_num": bottleneck_num,
        "batch_size": batch_size,
        "cuda_device": cuda_device,
        "max_epoch": max_epoch,
        "num_workers": num_workers,
        "opt": opt,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "lr": lr,
        "lr_scheduler": lr_scheduler,
        "gamma": gamma,
        "steps": [steps_start, steps_end],
        "middle_epoch": middle_epoch,
        "distance_option": distance_option,
        "distance_loss": distance_loss,
        "distance_tradeoff": distance_tradeoff,
        "distance_lambda": distance_lambda,
        "adversarial_option": adversarial_option,
        "adversarial_loss": adversarial_loss,
        "hidden_size": hidden_size,
        "grl_option": grl_option,
        "grl_lambda": grl_lambda,
        "adversarial_tradeoff": adversarial_tradeoff,
        "adversarial_lambda": adversarial_lambda,
        "wavelet": wavelet,
    }
    # 这里更新参数
    all_params = update_args_param(args, **args_params_dict)
    # 这里进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.strip()
    warnings.filterwarnings("ignore")
    save_dir = os.path.join(args.checkpoint_dir, args.model_name + "_" + datetime.strftime(datetime.now(), "%m%d-%H%M%S"))
    setattr(args, "save_dir", save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # 设定日志
    setlogger(os.path.join(args.save_dir, "train.log"))
    # 保存超参数
    for k, v in args.__dict__.items():
        if k[-3:] != "dir":
            logging.info(f"{k}: {v}")
    # 训练
    trainer = train_utils(args)
    trainer.setup()
    trainer.train()
    fig = trainer.generate_fig()
    # 这里返回各种结果
    return all_params, fig


# 下面是信号推理的函数
def signal_inference(model_file, signal_file):
    if model_file is None or signal_file is None:
        raise ValueError("请上传模型文件和信号数据！")
    model_state_dict = torch.load(model_file)
    if isinstance(signal_file, list):
        for signal_file_single in signal_file:
            signal = pd.read_csv(signal_file_single)
    else:
        signal = pd.read_csv(signal_file)
    result = predict(model_state_dict, signal)
    return result


def change_source_split(source_config_radio):
    source_splits = conditions[source_config_radio]
    return gr.update(choices=source_splits, value=source_splits[0])


def change_target_split(target_config_radio):
    target_splits = conditions[target_config_radio]
    return gr.update(choices=target_splits, value=target_splits[0])


def change_bottleneck(bottleneck):
    return gr.update(visible=bottleneck)


def change_opt(opt):
    if opt == "sgd":
        return gr.update(visible=True), gr.update(visible=True)
    elif opt == "adam":
        return gr.update(visible=False), gr.update(visible=False)


def change_lr_scheduler(lr_scheduler):
    if lr_scheduler == "step":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    elif lr_scheduler == "exp":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    elif lr_scheduler == "stepLR":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    elif lr_scheduler == "fix":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def change_steps_start(steps_start, steps_end):
    if steps_start >= steps_end:
        steps_start = steps_end - 1
    return gr.update(value=steps_start, maximum=steps_end - 1)


def change_steps_end(steps_start, steps_end):
    if steps_end <= steps_start:
        steps_end = steps_start + 1
    return gr.update(value=steps_end, minimum=steps_start + 1)


def change_middle_epoch(max_epoch, middle_epoch):
    if middle_epoch >= max_epoch:
        middle_epoch = max_epoch - 1
    return gr.update(value=middle_epoch, maximum=max_epoch - 1)


def change_distance_option(distance_option, distance_tradeoff):
    return gr.update(visible=distance_option), gr.update(visible=distance_option), gr.update(visible=(distance_option and distance_tradeoff == "Cons"))


def change_adversarial_option(adversarial_option, adversarial_tradeoff):
    return (
        gr.update(visible=adversarial_option),
        gr.update(visible=adversarial_option),
        gr.update(visible=adversarial_option),
        gr.update(visible=adversarial_option),
        gr.update(visible=adversarial_option),
        gr.update(visible=(adversarial_option and adversarial_tradeoff == "Cons")),
    )


def change_distance_tradeoff(distance_option, distance_tradeoff):
    return gr.update(visible=(distance_option and distance_tradeoff == "Cons"))


def change_adversarial_tradeoff(adversarial_option, adversarial_tradeoff):
    return (gr.update(visible=(adversarial_option and adversarial_tradeoff == "Cons")),)


# 创建一个绘图函数
def create_plot():
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    fig, ax = plt.subplots()
    ax.plot(x, y, label="y = x^2")
    ax.set_title("示例折线图")
    ax.set_xlabel("X 轴")
    ax.set_ylabel("Y 轴")
    ax.legend()
    return fig


# gradio BFDS_web.py --demo-name app
with gr.Blocks(title="BFDS WebUI") as app:
    with gr.Tab("模型训练"):
        gr.Markdown("在此模块中，您可以选择不同的迁移学习方法进行模型训练。")
        with gr.Row():
            with gr.Column():
                source_config_radio = gr.Radio(
                    label="选择源域数据集名称",
                    choices=list(conditions.keys()),
                    value=args.transfer_task[0][0],
                )
                source_split_radio = gr.Radio(
                    label="选择源域数据集工况",
                    choices=conditions[args.transfer_task[0][0]],
                    value=args.transfer_task[0][1],
                )
                target_config_radio = gr.Radio(
                    label="选择目标域数据集名称",
                    choices=list(conditions.keys()),
                    value=args.transfer_task[1][0],
                )
                target_split_radio = gr.Radio(
                    label="选择目标域数据集工况",
                    choices=conditions[args.transfer_task[1][0]],
                    value=args.transfer_task[1][1],
                )
                normalize_type_radio = gr.Radio(
                    label="选择归一化方式",
                    choices=["mean-std", "min-max", None],
                    value=args.normalize_type,
                )
                model_name_radio = gr.Radio(
                    label="选择模型名称",
                    choices=["CNN"],
                    value=args.model_name,
                )
                bottleneck_checkbox = gr.Checkbox(
                    label="是否使用瓶颈层",
                    value=args.bottleneck,
                )
                bottleneck_num_slider = gr.Slider(1, 1024, label="瓶颈层神经元个数", step=1, value=args.bottleneck_num, visible=args.bottleneck)
                batch_size_slider = gr.Slider(1, 258, label="batch_size", step=1, value=args.batch_size)
                cuda_device_radio = gr.Radio(
                    label="选择GPU设备",
                    choices=["0"],
                    value=args.cuda_device,
                )
                max_epoch_slider = gr.Slider(args.middle_epoch + 1, 100, label="max_epoch", step=1, value=args.max_epoch)
                num_workers_slider = gr.Slider(1, 16, label="num_workers", step=1, value=args.num_workers)
                opt_radio = gr.Radio(
                    label="选择优化器",
                    choices=["sgd", "adam"],
                    value=args.opt,
                )
                momentum_slider = gr.Slider(0, 1, label="momentum", step=0.01, value=args.momentum)
                weight_decay_slider = gr.Slider(1e-5, 1e-1, label="weight_decay", step=1e-5, value=args.weight_decay)
                lr_slider = gr.Slider(1e-5, 1e-2, label="学习率", step=1e-5, value=args.lr)
                lr_scheduler_radio = gr.Radio(
                    label="学习率调度器",
                    choices=["step", "exp", "stepLR", "fix"],
                    value=args.lr_scheduler,
                )
                gamma_slider = gr.Slider(1e-5, 1e-2, label="gamma", step=1e-5, value=args.gamma, visible=args.lr_scheduler != "fix")
                steps_start_slider = gr.Slider(1, args.steps[1] - 1, label="steps 第一个值", step=1, value=args.steps[0], visible=(args.lr_scheduler == "step" or args.lr_scheduler == "stepLR"))
                steps_end_slider = gr.Slider(args.steps[0] + 1, 1000, label="steps 第二个值", step=1, value=args.steps[1], visible=(args.lr_scheduler == "step" or args.lr_scheduler == "stepLR"))
                middle_epoch_slider = gr.Slider(1, args.max_epoch - 1, label="middle_epoch", step=1, value=args.middle_epoch)
                wavelet_radio = gr.Radio(
                    label="选择波形变换",
                    choices=["cmor1.5-1.0"],
                    value=args.wavelet,
                )
            with gr.Column():
                distance_option_checkbox = gr.Checkbox(
                    label="是否使用距离损失",
                    value=args.distance_option,
                )
                distance_loss_radio = gr.Radio(label="距离损失函数", choices=["MK-MMD", "JMMD", "CORAL"], value="MK-MMD", visible=args.distance_option)
                distance_tradeoff_radio = gr.Radio(label="距离损失权重", choices=["Cons", "Step"], value=args.distance_tradeoff, visible=args.distance_option)
                distance_lambda_slider = gr.Slider(1, 2, label="距离损失权重", step=1e-5, value=args.distance_lambda, visible=(args.distance_option and args.distance_tradeoff == "Cons"))
                adversarial_option_checkbox = gr.Checkbox(
                    label="是否使用对抗损失",
                    value=args.adversarial_option,
                )
                adversarial_loss_radio = gr.Radio(label="对抗损失函数", choices=["MK-MMD", "JMMD", "CORAL"], value="MK-MMD", visible=args.adversarial_option)
                hidden_size_slider = gr.Slider(1, 1024, label="对抗层神经元个数", step=1, value=args.hidden_size, visible=args.adversarial_option)
                grl_option_radio = gr.Radio(label="是否使用梯度反转层", choices=["Step"], value=args.grl_option, visible=args.adversarial_option)
                grl_lambda_slider = gr.Slider(1, 2, label="梯度反转层系数", step=1e-5, value=args.grl_lambda, visible=args.adversarial_option)
                adversarial_tradeoff_radio = gr.Radio(label="对抗损失权重", choices=["Cons", "Step"], value=args.adversarial_tradeoff, visible=args.adversarial_option)
                adversarial_lambda_slider = gr.Slider(1, 2, label="对抗损失权重", step=1e-5, value=args.adversarial_lambda, visible=(args.adversarial_option and args.adversarial_tradeoff == "Cons"))

        transfer_learning_button = gr.Button("开始训练")
        with gr.Row():
            with gr.Column():
                args_all_params = gr.Textbox(label="更新结果", lines=8)
            with gr.Column():
                plot_component = gr.Plot(label="训练结果图表")

    with gr.Tab("信号推理"):
        model_file = gr.File(label="模型文件", file_count="single", file_types=[".bin", ".pth", ".pt"])
        with gr.Tab("单次推理"):
            gr.Markdown("在此模块中，您可以上传信号数据进行推理。")
            signal_file_single = gr.File(label="上传信号数据", file_count="single", file_types=[".csv"])
            signal_inference_single_button = gr.Button("开始推理")
            signal_inference_single_output = gr.Textbox(label="推理结果", lines=8)
        with gr.Tab("批量推理"):
            gr.Markdown("在此模块中，您可以上传信号数据进行批量推理。")
            signal_file_multiple = gr.File(label="上传信号数据", file_count="multiple", file_types=[".csv"])
            signal_inference_multiple_button = gr.Button("开始批量推理")
            signal_inference_multiple_output = gr.Textbox(label="批量推理结果", lines=8)

    # 下面是所有函数绑定
    transfer_learning_button.click(
        transfer_learning,
        inputs=[
            source_config_radio,
            source_split_radio,
            target_config_radio,
            target_split_radio,
            normalize_type_radio,
            model_name_radio,
            bottleneck_checkbox,
            bottleneck_num_slider,
            batch_size_slider,
            cuda_device_radio,
            max_epoch_slider,
            num_workers_slider,
            opt_radio,
            momentum_slider,
            weight_decay_slider,
            lr_slider,
            lr_scheduler_radio,
            gamma_slider,
            steps_start_slider,
            steps_end_slider,
            middle_epoch_slider,
            distance_option_checkbox,
            distance_loss_radio,
            distance_tradeoff_radio,
            distance_lambda_slider,
            adversarial_option_checkbox,
            adversarial_loss_radio,
            hidden_size_slider,
            grl_option_radio,
            grl_lambda_slider,
            adversarial_tradeoff_radio,
            adversarial_lambda_slider,
            wavelet_radio,
        ],
        outputs=[args_all_params, plot_component],
    )
    # ============================================================================================================================
    source_config_radio.change(change_source_split, inputs=[source_config_radio], outputs=[source_split_radio])
    target_config_radio.change(change_target_split, inputs=[target_config_radio], outputs=[target_split_radio])
    opt_radio.change(change_opt, inputs=[opt_radio], outputs=[momentum_slider, weight_decay_slider])
    bottleneck_checkbox.change(change_bottleneck, inputs=[bottleneck_checkbox], outputs=[bottleneck_num_slider])
    lr_scheduler_radio.change(change_lr_scheduler, inputs=[lr_scheduler_radio], outputs=[steps_start_slider, steps_end_slider, gamma_slider])
    steps_start_slider.change(change_steps_start, inputs=[steps_start_slider, steps_end_slider], outputs=[steps_start_slider])
    steps_end_slider.change(change_steps_end, inputs=[steps_start_slider, steps_end_slider], outputs=[steps_end_slider])
    max_epoch_slider.change(change_middle_epoch, inputs=[max_epoch_slider, middle_epoch_slider], outputs=[middle_epoch_slider])
    distance_option_checkbox.change(change_distance_option, inputs=[distance_option_checkbox, distance_tradeoff_radio], outputs=[distance_loss_radio, distance_tradeoff_radio, distance_lambda_slider])
    adversarial_option_checkbox.change(
        change_adversarial_option,
        inputs=[adversarial_option_checkbox, adversarial_tradeoff_radio],
        outputs=[adversarial_loss_radio, hidden_size_slider, grl_option_radio, grl_lambda_slider, adversarial_tradeoff_radio, adversarial_lambda_slider],
    )
    distance_tradeoff_radio.change(change_distance_tradeoff, inputs=[distance_option_checkbox, distance_tradeoff_radio], outputs=[distance_lambda_slider])
    adversarial_tradeoff_radio.change(change_adversarial_tradeoff, inputs=[adversarial_option_checkbox, adversarial_tradeoff_radio], outputs=[adversarial_lambda_slider])
    # ============================================================================================================================
    signal_inference_single_button.click(signal_inference, inputs=[model_file, signal_file_single], outputs=signal_inference_single_output)
    signal_inference_multiple_button.click(signal_inference, inputs=[model_file, signal_file_multiple], outputs=signal_inference_multiple_output)

app.queue()
app.launch()
