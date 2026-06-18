# Football Player and Ball Detection (YOLO26)

A real-time vision detection pipeline designed for football (soccer) analytics. This repository utilizes the state-of-the-art **YOLO26** model family to detect players and the football in match images.

---

## 🚀 Why YOLO26?

**YOLO26** (specifically `yolo26m`) represents a massive step forward in object detection compared to prior generations:

1. **End-to-End, NMS-Free Inference:** By introducing native end-to-end inference, YOLO26 removes the need for Non-Maximum Suppression (NMS). This reduces overall processing pipeline complexity and latency.
2. **Small-Target-Awareness (STAL):** Incorporates advanced label assignment strategies specifically optimized to detect small objects. This is a massive advantage in sports analytics where the football/soccer ball is tiny compared to the rest of the pitch.
3. **Lighter, Faster Architecture:** The model removes Distribution Focal Loss (DFL) to streamline the regression head, yielding up to **43% faster CPU inference** compared to older architectures.
4. **Stable Training with MuSGD:** Uses a hybrid MuSGD optimizer to stabilize gradient updates and accelerate training convergence.

---

## 📁 Repository Structure

* `train.py` - Script used to configure and trigger training of `yolo26m` on the custom Football Player dataset using CUDA/GPU acceleration.
* `predict.py` - A basic inference script that loads trained weights to perform predictions on standard images.
* `predict2.py` - An enhanced, robust inference script that features:
  * **Dynamic Output Naming:** Saves prediction results matching the input name (e.g., `save.jpeg` saves to `save_prediction.jpg`).
  * **Pillow AVIF Plugin Support:** Automatically handles modern AVIF images (e.g., `netherlands-v-japan-group-f-fifa-world-cup-2026.avif`) by registering the `pillow_avif` plugin wrapper.
  * **Adjustable confidence settings.**

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Sanvith6/football-player-detection-yolo.git
   cd football-player-detection-yolo
   ```

2. **Install Dependencies:**
   Make sure you have PyTorch installed with GPU support if you intend to train.
   ```bash
   pip install ultralytics pillow pillow-avif-plugin
   ```

---

## 📊 Running Predictions

To run player and ball detection on your images using the advanced prediction script:

1. Open `predict2.py` and set your `input_image` path.
2. Execute the script:
   ```bash
   python predict2.py
   ```

The script will automatically run inference, print the detection details (classes, confidence, and bounding box coordinates) to the terminal, save the annotated result image, and show it.

### Example Run Outputs

#### Detections on `save.jpeg` (7 objects):
* **FOOTBALL** | Confidence: **91%** | Bounding Box: `[107.9, 128.4, 120.7, 141.6]`
* **PLAYER** | Confidence: **86%** | Bounding Box: `[125.3, 77.0, 176.5, 146.8]`
* **PLAYER** | Confidence: **83%** | Bounding Box: `[227.3, 65.7, 266.5, 131.4]`
* **PLAYER** | Confidence: **60%** | Bounding Box: `[18.7, 80.9, 110.8, 152.8]`
* ...and more.

Result saved dynamically as `save_prediction.jpg`.
