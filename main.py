import gradio as gr
import matplotlib
import matplotlib.pyplot as plt

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
# 我们可以把基础模型上传到hugging face的模型库中，这样就可以直接使用了


# 下面是信号推理的函数
def signal_inference():
    pass


# 下面是模型训练的函数
def transfer_learning():
    pass


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
    with gr.Tab("信号推理"):
        gr.Markdown("在此模块中，您可以对输入信号进行推理分析。")
        gr.File(
            label="上传已经训练好的模型文件",
            file_count="single",
            file_types=[".pt"],
        )
        with gr.Tab("单次推理"):
            gr.Markdown("上传单个信号文件，系统将对其进行推理并返回结果。")
            with gr.Row():
                with gr.Column():
                    gr.File(
                        label="上传信号文件",
                        file_count="single",
                        file_types=[".csv"],
                    )
                with gr.Column():
                    gr.Plot(create_plot)

        with gr.Tab("批量推理"):
            gr.Markdown("上传多个信号文件，系统将批量处理并返回所有推理结果。")
            with gr.Row():
                with gr.Column():
                    gr.File(
                        label="上传信号文件",
                        file_count="multiple",
                        file_types=[".csv"],
                    )
                with gr.Column():
                    gr.Plot(create_plot)
    with gr.Tab("模型训练"):
        gr.Markdown("在此模块中，您可以选择不同的迁移学习方法进行模型训练。")

        with gr.Tab("基于映射"):
            gr.Markdown("使用基于特征映射的迁移学习方法训练模型。")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("左侧内容")
                    gr.Checkbox(
                        label="是否使用预训练模型",
                    )
                with gr.Column():
                    gr.Markdown("右侧内容")
                    gr.Plot(create_plot)
        with gr.Tab("基于领域对抗"):
            gr.Markdown("使用基于领域对抗的迁移学习方法训练模型。")

app.queue()
app.launch()
# gradio main.py --demo-name app
