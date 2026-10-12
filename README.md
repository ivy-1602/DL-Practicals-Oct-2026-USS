# Deeeeeeep Learning Diaries
_Eleven practicals. Too many neural networks. Somehow, they all run._

---

## What's This?

My Third Year Deep Learning Lab-2 work: 11 practicals that took me from _"what is backpropagation"_ to putting a model behind a REST API.

Basically: _"What if we add another layer?"_ Repeat until it works.

Each practical is a small case study. Some involve images. Some involve text. One predicts stock prices. One translates English to French. One has an endpoint now.

---

## 🗂️ Quick Look

```
Deep-Learning-Lab-2/
├── 01_Backpropagation_Multilayer_Neural_Networks/   → MNIST + Deep MLP
├── 02_MLP_Activation_Functions/                     → Sigmoid, Tanh, ReLU, ELU
├── 03_Gradient_Problems_Regularization/             → Dropout, L1/L2, Normalization
├── 04_CNN_Architectures_Image_Classification/       → LeNet, AlexNet, VGG16, ResNet50
├── 05_Real-Time_Object_Detection/                   → R-CNN, YOLO, SSD MobileNet
├── 06_Semantic_Segmentation_U-Net/                  → Pixel-wise segmentation
├── 07_RNN_GRU_Stock_Forecasting/                    → RNN vs GRU
├── 08_English-to-French_Seq2Seq_Translation/        → Encoder-Decoder + LSTM
├── 09_MNIST_Digit_Classification_REST_API/          → Neural net + Flask
├── 10_Named_Entity_Recognition/                     → spaCy NER
└── 11_LLM_ViT_ASR_CI-CD_Workflow/                   → LLMs, ViT, ASR, CI/CD
```

---

## 👀 What's Inside

| #  | Practical | Concept | Dataset | The Vibe |
|----|-----------|---------|---------|----------|
| 1  | Backpropagation | Deep MLP | MNIST | Starting from the beginning |
| 2  | Activation Functions | Sigmoid, Tanh, ReLU, ELU | IRIS | Pick your non-linearity |
| 3  | Gradients & Regularization | Dropout, L1/L2 | Fashion-MNIST | Please don't explode |
| 4  | CNN Architectures | Transfer Learning | CIFAR-10 | Image era begins |
| 5  | Object Detection | R-CNN, YOLO, SSD | Pascal VOC | I see objects everywhere |
| 6  | Segmentation | U-Net | Oxford-IIIT Pet | Every pixel has a job |
| 7  | RNN & GRU Forecasting | Sequence models | Yahoo Finance | Predicting the future, allegedly |
| 8  | English → French | Seq2Seq + LSTM | English-French Pairs | Bonjour, neural network |
| 9  | Digit Classifier API | Flask REST API | MNIST | The model has an address now |
| 10 | Named Entity Recognition | spaCy | Entity Annotated Corpus | PERSON. ORG. LOCATION. Done. |
| 11 | LLMs, ViT, ASR, CI/CD | Modern AI pipelines | Multiple | The syllabus said "everything" |

---

## Running the Code

```bash
pip install numpy matplotlib tensorflow
```

Extras: `ultralytics` for object detection, `spacy` + `python -m spacy download en_core_web_sm` for NER.

Then:

```bash
python your_file.py
```

Some files save their graphs as `.png` automatically.
**Run → Train → Plot → Panic → Debug → Run again.**

---

## The Real Learning

- A model can have millions of parameters and still be confidently wrong.
- More layers ≠ automatically better.
- Debugging tensor shapes is a lifestyle.
- Stock prediction is not a crystal ball.
- A model inside a `.keras` file is not an application. APIs are where it meets the world.

---

## The Datasets

Standard SPPU Lab-2 datasets (MNIST, IRIS, CIFAR-10, Pascal VOC and friends). Some demos use smaller or generated data, because downloading a dataset can take longer than training the model.

---

## 📧 Contact

**Uma Salunke** — AI & ML Engineering Student

[LinkedIn](https://linkedin.com/in/umasalunke7) • [GitHub](https://github.com/ivy-1602) • [Email](mailto:umasalunke7@gmail.com)

---

## 📄 License

MIT License — use it, learn from it, and please don't submit the exact same code as your own lab work. Your professor has probably seen it before.

---

## 📝 Final Note

This repo isn't a textbook. It's a record of implementing the things that looked terrifying in the syllabus.

Some graphs look beautiful. Some look like the model gave up. Every working output felt disproportionately satisfying.

_The best model is the one that runs._
_The second best is the one you can explain to your professor._

Made with 🧠, 💻 & a questionable number of `print()` statements.

---

P.S. — If the model says it's 99% confident, remember: confidence and correctness are two very different things.
