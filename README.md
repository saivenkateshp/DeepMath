# DeepMath: Handwritten Math Expression Recognition

A computer-vision pipeline that **reads a handwritten arithmetic expression from an image and evaluates it**. OpenCV segments the expression into individual symbols, a **CNN (TensorFlow/Keras)** classifies each symbol, and the recognized expression is then computed.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

---

## How it works

```
input image ─▶ grayscale ─▶ edge detection + contours (OpenCV, imutils) ─▶ crop each symbol, sorted left→right
            ─▶ resize to 32×32, normalize ─▶ CNN classifier (20 classes) ─▶ rebuild expression string ─▶ evaluate
```

### 1. Dataset
- **188,445 grayscale images** of handwritten math symbols in **20 classes**: digits `0–9`, operators `+ - times div`, and brackets `( ) [ ] { }`.
- 80/20 train/validation split: **150,764** training and **37,681** validation images.
- Augmentation with `ImageDataGenerator` (rescaling, shear, zoom). The notebook also plots the number of images per class to check for class imbalance.

### 2. Model: CNN in TensorFlow/Keras

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU, input 32×32×1 |
| MaxPool2D | 2×2 |
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPool2D | 2×2 |
| Dropout | 0.2 |
| Flatten → Dense | 128 units, ReLU |
| Dense (output) | 20 units, softmax |

Trained with the **Adam** optimizer and **categorical cross-entropy** loss for 15 epochs. The model has about **160K parameters**.

### 3. Results

| Metric (epoch 15) | Value |
|---|---|
| Training accuracy | **96.6%** |
| Validation accuracy | **97.5%** |
| Validation loss | 0.078 |

The notebook plots training and validation accuracy and loss for each epoch.

### 4. Expression recognition
For a full expression image (for example `exp1.JPG`), the pipeline:
1. converts the image to grayscale and finds symbol contours with OpenCV,
2. crops each symbol, sorts them left to right, and resizes them to 32×32,
3. classifies each symbol with the CNN and maps class indices back to characters,
4. joins the characters into an expression and evaluates it, for example `157914*13 = 2052882`.

The trained model is saved and reloaded with `tf.keras.models.save_model` / `load_model`.

---

## Repository contents

| File | Description |
|---|---|
| `DeepMath.ipynb` | End-to-end notebook: data loading, augmentation, CNN training, evaluation plots, and expression recognition |
| `Presentation.pptx` | Project presentation slides |

## Running it

```bash
pip install tensorflow opencv-python imutils scikit-learn seaborn matplotlib pandas numpy
jupyter notebook DeepMath.ipynb
```

The notebook expects the symbol dataset as `dataset.zip` (extracted into `extracted_images/`, one folder per class) and test images such as `exp1.JPG` in the working directory. These are not included because of their size. The notebook was originally run on Google Colab (Python 3.7).

---

## Tech stack

**Python** · **TensorFlow / Keras** · **OpenCV** · **imutils** · **NumPy / Pandas** · **Matplotlib / Seaborn** · **scikit-learn**
