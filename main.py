import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
from utils.args import Argument, update_param
import pandas as pd
import torch
from utils.predict import predict

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
def transfer_learning(batch_size, optimizer, learning_rate, scheduler, transfer_method, distance_loss):
    # 这里更新参数
    all_params = update_param(args, batch_size, optimizer, learning_rate, scheduler, transfer_method, distance_loss)
    # 这里进行训练

    # 这里返回各种结果
    return all_params


# 下面是信号推理的函数
def signal_inference(model_file, signal_file):
    if model_file is None or signal_file is None:
        raise ValueError("请上传模型文件和信号数据！")
    model_state_dict = torch.load(model_file)
    if isinstance(signal_file, list):
        for signal_file_single in signal_file:
            signal = pd.read_csv(signal_file_single)
            # FIXME 最后做成(n,1,128)的形式
    else:
        signal = pd.read_csv(signal_file)
    result = predict(model_state_dict, signal)
    return result


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


with gr.Blocks(title="BFDS WebUI") as app:
    with gr.Tab("模型训练"):
        gr.Markdown("在此模块中，您可以选择不同的迁移学习方法进行模型训练。")
        with gr.Tab("使用预训练模型"):
            gr.Markdown("使用预训练模型进行迁移学习，您可以选择以下参数进行配置：")
            with gr.Row():
                with gr.Column():
                    batch_size_slider = gr.Slider(1, 258, label="batch_size", step=1, value=args.batch_size)
                    optimizer_radio = gr.Radio(
                        label="选择优化器",
                        choices=["Adam", "SGD", "RMSprop"],
                        value=args.opt.capitalize(),
                    )
                    learning_rate_slider = gr.Slider(1e-5, 1e-2, label="学习率", step=1e-5, value=args.lr)
                    scheduler_radio = gr.Radio(
                        label="学习率调度器",
                        choices=["step", "exp", "stepLR", "fix"],
                        value=args.lr_scheduler,
                    )
                    transfer_method_radio = gr.Radio(
                        label="迁移学习方式",
                        choices=["基于映射", "基于领域对抗"],
                        value="基于领域对抗" if args.adversarial_option else "基于映射",
                    )
                with gr.Column():
                    distance_loss_radio = gr.Radio(
                        label="距离损失函数",
                        choices=["MK-MMD", "JMMD", "CORAL"],
                        value="MK-MMD",  # 修复默认值为有效选项
                    )
            update_button = gr.Button("开始训练")
            with gr.Row():
                with gr.Column():
                    # FIXME 需要弄好看一点
                    args_all_params = gr.Textbox(label="更新结果", lines=8)
                with gr.Column():
                    gr.Plot(create_plot)
        with gr.Tab("不使用预训练模型"):
            gr.Markdown("使用从零开始训练的方式，不依赖预训练模型。")

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
    update_button.click(
        transfer_learning,
        inputs=[batch_size_slider, optimizer_radio, learning_rate_slider, scheduler_radio, transfer_method_radio, distance_loss_radio],
        outputs=args_all_params,
    )
    signal_inference_single_button.click(signal_inference, inputs=[model_file, signal_file_single], outputs=signal_inference_single_output)
    signal_inference_multiple_button.click(signal_inference, inputs=[model_file, signal_file_multiple], outputs=signal_inference_multiple_output)
app.queue()
app.launch()
