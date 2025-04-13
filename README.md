<div align="center">
  <img src="docs/favicon.png" alt="favicon" width="50">
  <h1>Bearing-Fault-Diagnosis-System-WebUI</h1>
</div>

![web-demo](/docs/web-demo.png)

---

## 快速部署

### 使用 CPU 版本

1. **拉取镜像**：

   ```bash
   docker pull ghcr.io/bfds-project/bearing-fault-diagnosis-system-webui:cpu1.0
   ```

2. **创建目录**：

   ```bash
   mkdir docker/cache
   mkdir docker/checkpoint
   ```

3. **运行容器**：

   - **CMD**:

     ```bash
     docker run -d -p 7860:7860 -v %cd%/docker/cache:/app/cache -v %cd%/docker/checkpoint:/app/checkpoint --name bfds_cpu Bearing-Fault-Diagnosis-System-WebUI-CPU
     ```

   - **PowerShell**:

     ```bash
     docker run -d -p 7860:7860 -v ${PWD}/docker/cache:/app/cache -v ${PWD}/docker/checkpoint:/app/checkpoint --name bfds_cpu Bearing-Fault-Diagnosis-System-WebUI-CPU
     ```

4. **启动后访问**：[http://127.0.0.1:7860](http://127.0.0.1:7860)

---

### 使用 GPU 版本

1. **拉取镜像**：

   ```bash
   docker pull ghcr.io/bfds-project/bearing-fault-diagnosis-system-webui:gpu1.0
   ```

2. **创建目录**：

   ```bash
   mkdir docker/cache
   mkdir docker/checkpoint
   ```

3. **运行容器**：

   - **CMD**:

     ```bash
     docker run -d --gpus all -p 7860:7860 -v %cd%/docker/cache:/app/cache -v %cd%/docker/checkpoint:/app/checkpoint --name bfds_gpu Bearing-Fault-Diagnosis-System-WebUI-GPU
     ```

   - **PowerShell**:

     ```bash
     docker run -d --gpus all -p 7860:7860 -v ${PWD}/docker/cache:/app/cache -v ${PWD}/docker/checkpoint:/app/checkpoint --name bfds_gpu Bearing-Fault-Diagnosis-System-WebUI-GPU
     ```

4. **启动后访问**：[http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 自行部署镜像

### 使用 CPU 版本

1. **构建镜像**：

   ```bash
   docker build -f Dockerfile.cpu -t Bearing-Fault-Diagnosis-System-WebUI-CPU .
   ```

2. **创建目录**：

   ```bash
   mkdir docker/cache
   mkdir docker/checkpoint
   ```

3. **运行容器**：

   - **CMD**:

     ```bash
     docker run -d -p 7860:7860 -v %cd%/docker/cache:/app/cache -v %cd%/docker/checkpoint:/app/checkpoint --name bfds_cpu Bearing-Fault-Diagnosis-System-WebUI-CPU
     ```

   - **PowerShell**:

     ```bash
     docker run -d -p 7860:7860 -v ${PWD}/docker/cache:/app/cache -v ${PWD}/docker/checkpoint:/app/checkpoint --name bfds_cpu Bearing-Fault-Diagnosis-System-WebUI-CPU
     ```

4. **启动后访问**：[http://127.0.0.1:7860](http://127.0.0.1:7860)

---

### 使用 GPU 版本

1. **构建镜像**：

   ```bash
   docker build -f Dockerfile.gpu -t Bearing-Fault-Diagnosis-System-WebUI-GPU .
   ```

2. **创建目录**：

   ```bash
   mkdir docker/cache
   mkdir docker/checkpoint
   ```

3. **运行容器**：

   - **CMD**:

     ```bash
     docker run -d --gpus all -p 7860:7860 -v %cd%/docker/cache:/app/cache -v %cd%/docker/checkpoint:/app/checkpoint --name bfds_gpu Bearing-Fault-Diagnosis-System-WebUI-GPU
     ```

   - **PowerShell**:

     ```bash
     docker run -d --gpus all -p 7860:7860 -v ${PWD}/docker/cache:/app/cache -v ${PWD}/docker/checkpoint:/app/checkpoint --name bfds_gpu Bearing-Fault-Diagnosis-System-WebUI-GPU
     ```

4. **启动后访问**：[http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 调试模式启动

1. **克隆项目**：

   ```bash
   git clone https://github.com/BFDS-Project/Bearing-Fault-Diagnosis-System-WebUI.git
   cd Bearing-Fault-Diagnosis-System-WebUI
   ```

2. **创建虚拟环境**：

   - **CPU 版本**：

     ```bash
     conda create -n BFDSWeb-cpu python=3.13
     conda activate BFDSWeb-gpu
     pip install -r requirements-cpu.txt
     ```

   - **GPU 版本**：

     ```bash
     conda create -n BFDSWeb-gpu python=3.13
     conda activate BFDSWeb-gpu
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     pip install -r requirements-gpu.txt
     ```
     
     > 想要下载更新的版本，请访问 [PyTorch 官网](https://pytorch.org/)。

4. **运行迁移学习验证（可选）**：

   ```bash
   python BFDS_train.py
   ```

5. **启动 Gradio 前端网页**：

   ```bash
   python BFDS_web.py
   ```

6. **启动后访问**：[http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 相关资源

- **Hugging Face 数据集**：[https://huggingface.co/datasets/BFDS-Project/Bearing-Fault-Diagnosis-System](https://huggingface.co/datasets/BFDS-Project/Bearing-Fault-Diagnosis-System)