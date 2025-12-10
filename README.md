# Banknote Classification System Using Convolutional Neural Networks (CNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📄 Project Overview
This project is a **Deep Learning classification system** designed to distinguish between **genuine and counterfeit banknotes** with high accuracy. Developed as a **Final Year Project (FYP)** at **Universiti Tun Hussein Onn Malaysia (UTHM)**[cite: 13, 56], the system utilizes a **Convolutional Neural Network (CNN)** to analyze banknote images and identify forgery patterns that are often invisible to the naked eye.

The model is built using **TensorFlow/Keras** and utilizes **OpenCV** for image preprocessing, demonstrating the practical application of AI in financial security.

## 🛠️ Tech Stack
* **Language:** Python
* [cite_start]**Deep Learning Framework:** TensorFlow (Keras)
* [cite_start]**Computer Vision:** OpenCV 
* [cite_start]**Data Processing & Evaluation:** Scikit-learn, NumPy, Pandas, Matplotlib 

## 📊 Dataset
* **Source:** https://universe.roboflow.com/fyp-leirw/malaysian-banknote , https://universe.roboflow.com/malaysia-currency-detector/currency-detector-actna
* **Description:** The dataset consists of 10374 images/samples of banknotes.
* **Preprocessing:** Images were resized, normalized to improve model generalization.

## 🧠 Model Architecture
The solution implements a **Convolutional Neural Network (CNN)**, which is highly effective for image classification tasks.
1.  **Input Layer:** Receives preprocessed banknote images.
2.  **Convolutional Layers:** Extract features (edges, textures, patterns) using filters.
3.  **Pooling Layers:** Reduce dimensionality to prevent overfitting and reduce computation.
4.  **Flatten Layer:** Converts 2D feature maps into a 1D vector.
5.  **Dense (Fully Connected) Layers:** Performs the final classification based on extracted features.
6.  **Output Layer:** Uses a 'Softmax' activation function to output the probability of the banknote being "Genuine" or "Counterfeit".

## 🚀 Installation & Setup
To run this project locally, follow these steps:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/](https://github.com/)[YourUsername]/banknote-classification-cnn.git
    cd banknote-classification-cnn
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage
1.  **Train the Model:**
    Run the training script to build and save the model.
    ```bash
    python CNN_model.py
    ```

2.  **Test the Model:**
    Run the prediction script on a sample image.
    ```bash
    python CNN_model.py --image_path "data/sample_banknote.jpg"
    ```

## 📈 Results
* **Training Accuracy:** [89.48%]
* **Validation Accuracy:** [92.09%]
* **Confusion Matrix:**

## 🔮 Future Improvements
* Deploy the model as a web app using **Streamlit** or **Flask**.
* Expand the dataset to include multi-currency detection.
* Implement real-time detection via webcam using OpenCV.

## 👥 Author & Acknowledgements
**Chok Wing Hong**
* **University:** Universiti Tun Hussein Onn Malaysia (UTHM)
* **Degree:** Bachelor of Science (Technology Mathematics) with Honours
* **Contact:** chalky1103@gmail.com

**Special Thanks:**
**Supervisor:** Mr. Lee Siaw Chong for guidance and supervision throughout the FYP.
