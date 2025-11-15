# Sign-Language-Recognition


Real-time American Sign Language Alphabet Detection using **MediaPipe Hand Landmarks** and a lightweight **MLP classifier** trained on webcam data.

---

## Overview

This project implements a real-time ASL alphabet recognition system using:

- **MediaPipe Hands** → extracts 21 hand landmarks per frame  
- **Custom MLP classifier** → classifies hand signs from normalized landmark vectors  
- **Webcam demo** → live prediction with drawing  
- **Local, personalized training data** → ensures high accuracy under your conditions

---

## Features

-  Real-time ASL alphabet prediction  
- Hand landmark visualization  
- Simple script to record labeled webcam samples  
- Lightweight MLP model, fast training

---

# Project Structure

```bash 
Sign-Language-Recognition/
├── README.md
├── Pipfile
├── Pipefile.lock
├── .gitignore
├── init.py
├── data/
│ ├── label_map.json
│ └── webcam_landmarks.npz
├── models/
│ └── mlp_webcam.pt
│ └── init.py
│ └── model_MLP.py
└── src/
├── init.py
├── dataset_asl_mnist.py
├── mediapipe_utils.py
├── record_webcam_samples.py
├── model_mlp.py
├── train_mlp.py
└── webcam_demo.py
└── test.py
```

# Usage

Install dependencies from the Pipfile

```bash
pipenv install
```


Run the demo using

``` bash
python -m src.webcam_demo
```