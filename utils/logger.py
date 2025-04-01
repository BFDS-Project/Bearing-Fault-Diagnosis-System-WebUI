import logging


def setlogger(path):
    """
    日志文件初始化
    Args:
        path(_str_): log文件保存路径
    """
    logger = logging.getLogger()
    if logger.hasHandlers():  # 检查是否已经有处理器
        logger.handlers.clear()  # 清除已有的处理器，避免重复输出

    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")  # 格式为  月-日 时：分：秒

    fileHandler = logging.FileHandler(path)  # 创建一个文件处理器，用于将日志写入指定文件
    fileHandler.setFormatter(logFormatter)  # 为文件处理器设置日志格式
    logger.addHandler(fileHandler)  # 将文件处理器添加到日志记录器中

    consoleHandler = logging.StreamHandler()  # 创建一个控制台处理器，用于将日志输出到控制台
    consoleHandler.setFormatter(logFormatter)  # 为控制台处理器设置日志格式
    logger.addHandler(consoleHandler)  # 将控制台处理器添加到日志记录器中
