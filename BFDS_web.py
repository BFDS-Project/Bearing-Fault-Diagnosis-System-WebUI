import logging
import os
import warnings
import zipfile
from datetime import datetime

import requests
import torch
import pandas as pd

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
    if not os.path.exists("cache"):
        os.makedirs("cache")
    os.environ["HUGGINGFACE_HUB_CACHE"] = "cache"

import gradio as gr
from BFDS_train import Argument
from utils.logger import setlogger
from utils.predict import predict
from utils.train import train_utils


# 初始化 Argument 实例
args = Argument()
args.set_recommended_params()


# 更新参数的函数
def transfer_learning(
    source_config,
    source_split,
    target_path,
    normalize_type,
    stratified_sampling,
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
):
    args_params_dict = {
        "transfer_task": [[source_config, source_split], []],
        "normalize_type": normalize_type,
        "stratified_sampling": stratified_sampling,
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
    }
    # 这里更新参数
    if target_path is None:
        raise ValueError("请上传目标域数据!")
    args.update_params(**args_params_dict)
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
    args.save_params(os.path.join(args.save_dir, "args.json"))
    # 训练
    trainer = train_utils(args, owned=True, data_path=target_path)
    trainer.setup()
    trainer.train()
    fig = trainer.generate_fig()

    # 压缩 save_dir 文件夹
    zip_filename = f"{trainer.save_dir}.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(trainer.save_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(trainer.save_dir, "..")))

    return fig, zip_filename


# 下面是信号推理的函数
def signal_inference(args_file, model_file, signal_file):
    if args_file is None or model_file is None or signal_file is None:
        raise ValueError("请上传参数文件和模型文件和信号数据!")
    args.load_params(args_file)
    result = []
    model_state_dict = torch.load(model_file)
    for signal_file_single in signal_file:
        prediction = predict(model_state_dict, signal_file_single, args)
        result.append({"文件名": signal_file_single, "预测值": prediction})
    return pd.DataFrame(result)


def change_source_split(source_config_radio):
    source_splits = args.conditions[source_config_radio]
    return gr.update(choices=source_splits, value=source_splits[0])


def change_bottleneck(bottleneck):
    return gr.update(interactive=bottleneck)


def change_opt(opt):
    if opt == "sgd":
        return gr.update(interactive=True), gr.update(interactive=True)
    elif opt == "adam":
        return gr.update(interactive=False), gr.update(interactive=False)


def change_lr_scheduler(lr_scheduler):
    if lr_scheduler == "step":
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
    elif lr_scheduler == "exp":
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True)
    elif lr_scheduler == "stepLR":
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
    elif lr_scheduler == "fix":
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)


def change_steps_start(steps_start):
    return gr.update(minimum=steps_start + 1)


def change_steps_end(steps_end):
    return gr.update(maximum=steps_end - 1)


def change_max_epoch(max_epoch):
    return gr.update(maximum=max_epoch - 1)


def change_middle_epoch(middle_epoch):
    return gr.update(minimum=middle_epoch + 1)


def change_distance_option(distance_option, distance_tradeoff):
    if distance_option:
        return gr.update(value=False), gr.update(interactive=distance_option), gr.update(interactive=distance_option), gr.update(interactive=(distance_option and distance_tradeoff == "Cons"))
    else:
        return gr.update(value=False), gr.update(interactive=distance_option), gr.update(interactive=distance_option), gr.update(interactive=(distance_option and distance_tradeoff == "Cons"))


def change_adversarial_option(adversarial_option, adversarial_tradeoff):
    return (
        gr.update(value=not adversarial_option),
        gr.update(interactive=adversarial_option),
        gr.update(interactive=adversarial_option),
        gr.update(interactive=adversarial_option),
        gr.update(interactive=adversarial_option),
        gr.update(interactive=adversarial_option),
        gr.update(interactive=(adversarial_option and adversarial_tradeoff == "Cons")),
    )


def change_distance_tradeoff(distance_option, distance_tradeoff):
    return gr.update(interactive=(distance_option and distance_tradeoff == "Cons"))


def change_adversarial_tradeoff(adversarial_option, adversarial_tradeoff):
    return (gr.update(interactive=(adversarial_option and adversarial_tradeoff == "Cons")),)


with open("docs/BFDS_font.html", "r", encoding="utf-8") as f:
    BFDS_font_html = f.read()

# gradio BFDS_web.py --demo-name app
with gr.Blocks(title="BFDS WebUI") as app:
    gr.HTML(BFDS_font_html)
    gr.Markdown("""
    # 轴承故障诊断系统
    基于深度迁移学习的智能轴承故障诊断系统。支持多种迁移学习算法、信号处理方法和故障诊断模型。
    """)
    with gr.Tab("模型训练"):
        gr.Markdown("在此模块中，您可以选择不同的迁移学习方法进行模型训练。")
        with gr.Row(equal_height=True):
            with gr.Column():
                source_config_radio = gr.Radio(
                    label="选择源域数据集名称",
                    choices=list(args.conditions.keys()),
                    value=args.transfer_task[0][0],
                )
                source_split_radio = gr.Radio(
                    label="选择源域数据集工况",
                    choices=args.conditions[args.transfer_task[0][0]],
                    value=args.transfer_task[0][1],
                )
                model_name_radio = gr.Radio(
                    label="选择模型名称",
                    choices=["CNN", "ResNet"],
                    value=args.model_name,
                )
            with gr.Column():
                target_file = gr.File(label="目标域数据集", file_count="single", file_types=[".csv"])
        with gr.Column():
            normalize_type_radio = gr.Radio(
                label="选择归一化方式",
                choices=["mean-std", "min-max", None],
                value=args.normalize_type,
            )
            cuda_device_radio = gr.Radio(
                label="选择GPU设备",
                choices=["0"],
                value=args.cuda_device,
            )
            stratified_sampling_checkbox = gr.Checkbox(
                label="是否启用分层抽样",
                value=args.stratified_sampling,
            )
            bottleneck_checkbox = gr.Checkbox(
                label="是否使用瓶颈层",
                value=args.bottleneck,
            )
            bottleneck_num_slider = gr.Slider(1, 1024, label="瓶颈层神经元个数", step=1, value=args.bottleneck_num, interactive=args.bottleneck)
        with gr.Row(equal_height=True):
            with gr.Column():
                batch_size_slider = gr.Slider(1, 258, label="batch_size", step=1, value=args.batch_size)
                max_epoch_slider = gr.Slider(args.middle_epoch + 1, 1000, label="max_epoch", step=1, value=args.max_epoch)
                num_workers_slider = gr.Slider(1, 16, label="num_workers", step=1, value=args.num_workers)
                opt_radio = gr.Radio(
                    label="选择优化器",
                    choices=["sgd", "adam"],
                    value=args.opt,
                )
                momentum_slider = gr.Slider(0, 1, label="momentum", step=0.01, value=args.momentum)
                weight_decay_slider = gr.Slider(1e-5, 1e-1, label="weight_decay", step=1e-5, value=args.weight_decay)
                lr_scheduler_radio = gr.Radio(
                    label="学习率调度器",
                    choices=["step", "exp", "stepLR", "fix"],
                    value=args.lr_scheduler,
                )
                lr_slider = gr.Slider(1e-5, 1e-2, label="学习率", step=1e-5, value=args.lr)
                gamma_slider = gr.Slider(1e-5, 1e-2, label="gamma", step=1e-5, value=args.gamma, interactive=args.lr_scheduler != "fix")
                steps_start_slider = gr.Slider(1, args.steps[1] - 1, label="steps 第一个值", step=1, value=args.steps[0], interactive=(args.lr_scheduler == "step" or args.lr_scheduler == "stepLR"))
                steps_end_slider = gr.Slider(args.steps[0] + 1, 1000, label="steps 第二个值", step=1, value=args.steps[1], interactive=(args.lr_scheduler == "step" or args.lr_scheduler == "stepLR"))
            with gr.Column():
                middle_epoch_slider = gr.Slider(0, args.max_epoch - 1, label="middle_epoch", step=1, value=args.middle_epoch)
                distance_option_checkbox = gr.Checkbox(
                    label="是否使用距离损失",
                    value=args.distance_option,
                )
                distance_loss_radio = gr.Radio(label="距离损失函数", choices=["MK-MMD", "JMMD", "CORAL"], value=args.distance_loss, interactive=args.distance_option)
                distance_tradeoff_radio = gr.Radio(label="距离损失权重", choices=["Cons", "Step"], value=args.distance_tradeoff, interactive=args.distance_option)
                distance_lambda_slider = gr.Slider(1, 2, label="距离损失权重", step=1e-5, value=args.distance_lambda, interactive=(args.distance_option and args.distance_tradeoff == "Cons"))
                adversarial_option_checkbox = gr.Checkbox(
                    label="是否使用对抗损失",
                    value=args.adversarial_option,
                )
                adversarial_loss_radio = gr.Radio(label="对抗损失函数", choices=["DA", "CDA", "CDA+E"], value=args.adversarial_loss, interactive=args.adversarial_option)
                hidden_size_slider = gr.Slider(1, 1024, label="对抗层神经元个数", step=1, value=args.hidden_size, interactive=args.adversarial_option)
                grl_option_radio = gr.Radio(label="是否使用梯度反转层", choices=["Step"], value=args.grl_option, interactive=args.adversarial_option)
                grl_lambda_slider = gr.Slider(1, 2, label="梯度反转层系数", step=1e-5, value=args.grl_lambda, interactive=args.adversarial_option)
                adversarial_tradeoff_radio = gr.Radio(label="对抗损失权重", choices=["Cons", "Step"], value=args.adversarial_tradeoff, interactive=args.adversarial_option)
                adversarial_lambda_slider = gr.Slider(
                    1, 2, label="对抗损失权重", step=1e-5, value=args.adversarial_lambda, interactive=(args.adversarial_option and args.adversarial_tradeoff == "Cons")
                )
        transfer_learning_button = gr.Button("开始训练")
        with gr.Row():
            with gr.Column():
                download_output = gr.File(label="下载训练结果压缩包", interactive=False)
            with gr.Column():
                plot_component = gr.Plot(label="训练结果图表")
    with gr.Tab("信号推理"):
        model_file = gr.File(label="模型文件", file_count="single", file_types=[".bin", ".pth", ".pt"])
        gr.Markdown("在此模块中，您可以上传信号数据进行批量推理。")
        signal_args_file = gr.File(label="参数文件", file_count="single", file_types=[".json"])
        signal_file_multiple = gr.File(label="上传信号数据", file_count="multiple", file_types=[".csv", ".xlsx", ".xls", ".txt", "wav", "mp3", "flac"])
        signal_inference_button = gr.Button("开始批量推理")
        signal_inference_Dataframe = gr.Dataframe(
            label="批量推理结果",
            headers=["文件名", "预测值"],
            datatype=["str", "list"],
            row_count=8,
            wrap=True,
            interactive=False,
        )
    # 下面是所有函数绑定
    transfer_learning_button.click(
        transfer_learning,
        inputs=[
            source_config_radio,
            source_split_radio,
            target_file,
            normalize_type_radio,
            stratified_sampling_checkbox,
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
        ],
        outputs=[plot_component, download_output],
    )
    source_config_radio.change(change_source_split, inputs=[source_config_radio], outputs=[source_split_radio])
    opt_radio.change(change_opt, inputs=[opt_radio], outputs=[momentum_slider, weight_decay_slider])
    bottleneck_checkbox.change(change_bottleneck, inputs=[bottleneck_checkbox], outputs=[bottleneck_num_slider])
    lr_scheduler_radio.change(change_lr_scheduler, inputs=[lr_scheduler_radio], outputs=[steps_start_slider, steps_end_slider, gamma_slider])
    steps_start_slider.change(change_steps_start, inputs=[steps_start_slider], outputs=[steps_end_slider])
    steps_end_slider.change(change_steps_end, inputs=[steps_end_slider], outputs=[steps_start_slider])
    max_epoch_slider.change(change_max_epoch, inputs=[max_epoch_slider], outputs=[middle_epoch_slider])
    middle_epoch_slider.change(change_middle_epoch, inputs=[middle_epoch_slider], outputs=[max_epoch_slider])
    distance_option_checkbox.change(
        change_distance_option, inputs=[distance_option_checkbox, distance_tradeoff_radio], outputs=[adversarial_option_checkbox, distance_loss_radio, distance_tradeoff_radio, distance_lambda_slider]
    )
    adversarial_option_checkbox.change(
        change_adversarial_option,
        inputs=[adversarial_option_checkbox, adversarial_tradeoff_radio],
        outputs=[distance_option_checkbox, adversarial_loss_radio, hidden_size_slider, grl_option_radio, grl_lambda_slider, adversarial_tradeoff_radio, adversarial_lambda_slider],
    )
    distance_tradeoff_radio.change(change_distance_tradeoff, inputs=[distance_option_checkbox, distance_tradeoff_radio], outputs=[distance_lambda_slider])
    adversarial_tradeoff_radio.change(change_adversarial_tradeoff, inputs=[adversarial_option_checkbox, adversarial_tradeoff_radio], outputs=[adversarial_lambda_slider])
    signal_inference_button.click(signal_inference, inputs=[signal_args_file, model_file, signal_file_multiple], outputs=signal_inference_Dataframe)

print("本地访问: http://127.0.0.1:7860")
print("Docker访问: http://127.0.0.1:7860")
app.queue()
app.launch(server_name="0.0.0.0", server_port=7860, favicon_path="docs/favicon.ico", show_error=True)
