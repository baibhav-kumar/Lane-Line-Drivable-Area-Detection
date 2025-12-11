This repository contains an enhanced implementation of **TwinLiteNet**, a state-of-the-art lightweight and real-time deep learning model for simultaneous **Drivable Area (DA) Segmentation** and **Lane Line (LL) Detection**.

This project focuses on architectural and training improvements to boost the performance of the original TwinLiteNet model while maintaining its high efficiency, making it suitable for deployment on edge devices like the NVIDIA Jetson series.
Drivable Area Segmentation - Identifies all navigable road surfaces
Lane Detection - Pinpoints precise lane boundaries
The model achieves 91.26% mIoU (Drivable Area) and 32.33% IoU (Lane Detection)
while maintaining only 0.379M parameters and running at 415 FPS on RTX A5000 and
60 FPS on Jetson Xavier NX.

Key Innovations over the base TwinLiteNet
✅
One shared encoder + two specialized decoders for multi-task learning
✅
Dual attention mechanism (Position & Channel Attention) for feature re nement
✅
Cross-Task Fusion (CFF) blocks enabling knowledge transfer between tasks
✅
Squeeze-and-Excitation (SE) blocks for intelligent channel recalibration
✅
Optimized attention kernels using PyTorch's 
scaled_dot_product_attention
✅
Depthwise-separable convolutions reducing parameters without sacri cing
accuracy


Model Version | DA mIoU (%) | LL IoU (%) | Improvement |
| :--- | :--- | :--- | :--- |
| Original TwinLiteNet | 90.6 | 30.0 | Baseline |
| **TwinLiteNet-Enhanced** | **91.01** | **31.79** | **+0.41 DA mIoU, +1.79 LL IoU** |
| **TwinLiteNet-Enhanced + TTA** | **91.26** | **32.33** | **+0.25 DA mIoU, +0.54 LL IoU** |


## 🛠️ Setup and Installation

### Prerequisites

* Python 3.8+
* PyTorch 1.7+

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/TwinLiteNet-Enhanced.git](https://github.com/your-username/TwinLiteNet-Enhanced.git)
    cd TwinLiteNet-Enhanced
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    conda create -n twinlite_env python=3.10
    conda activate twinlite_env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Dataset

This project uses the BDD100K segmentation dataset.

1.  Download the dataset.
2.  Set the `--data_root` argument in `train.py` to point to the root directory of the dataset.

## 🏃 Usage (Training)

The model can be trained by running `train.py` with command-line arguments.

```bash
# Example training command
python train.py \
    --data_root /path/to/bdd100k/ \
    --max_epochs 150 \
    --batch_size 12 \
    --lr 5e-4 \
    --ll_weight 3.5 \
    --savedir ./exp_enhanced
