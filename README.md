# 🏠 Predicting Kaggle's House Prices using a Multilayer Perceptron

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Kaggle-Dataset-blue?style=for-the-badge&logo=kaggle" />
  <img src="https://img.shields.io/badge/Task-Regression-orange?style=for-the-badge" />
</p>

---

## 📌 Part 1: Problem Introduction

### **1.1 Objective**
The main goal was building a **Multilayer Perceptron (MLP)** to estimate home sale prices using the Kaggle Houses dataset. This involved tackling a complex regression task with 79 varied input features. Unlike simple problems, this required careful treatment of gaps in entries, category transformation, and adjusting numerical ranges for stability.

### **1.2 Background Context**
This dataset includes **1,460 samples** for training, combining **7 numerical** alongside **52 categorical variables**. A major hurdle emerges from flawed inputs: around one-third of fields miss entries, and predictors span vastly different ranges—from tiny scores to huge floor areas.



### **1.3 Lab Objectives**
* 🏗️ Build and train an MLP to map property features to sale prices with minimal error.
* 🧪 Transform categories using **one-hot encoding** and handle missing numbers through imputation.
* 📈 Evaluate model output through **MSE** and graphical checks.
* 🔍 Spot weaknesses and suggest solid fixes for actual use.

---

## ⚙️ Part 2: Model Description

### **2.1 Approach**
The project uses a supervised learning approach with an MLP implemented in **PyTorch**. To stabilize learning, we adjusted the target by modeling $log(1+y)$ instead of actual prices, which improved optimization dynamics.

### **2.2 Model Architecture**
I built a **four-layer MLP** using a "wide and deep" setup:
* **Input Layer:** ~280 traits (after encoding).
* **Hidden Layer 1:** 512 neurons + Batch Normalization + Dropout (0.4).
* **Hidden Layer 2:** 256 neurons + Batch Normalization + Dropout (0.3).
* **Hidden Layer 3:** 128 neurons + Batch Normalization + Dropout (0.3).
* **Output Layer:** 1 neuron estimating log-price.



### **2.3 Key Components**
* **Activation:** `ReLU` is used in every hidden layer to avoid vanishing gradients.
* **Regularization:** `Dropout` (30-40% chance) was added to prevent overfitting by spreading learning across many pathways.
* **Batch Normalization:** Applied to scale inputs and make training more stable.

### **2.4 Training Process**
* **Loss Function:** Mean Squared Error (MSE).
* **Optimizer:** Adam Optimizer (Initial LR: 0.001).
* **Scheduler:** Reduces the learning rate by half every 100 epochs.
* **Execution:** 500 training cycles to ensure error levels settle.

---

## 📊 Part 3: Experimental Results

### **3.1 Loss Curves**

<img width="1000" height="500" alt="Image" src="https://github.com/user-attachments/assets/941a5a82-8dc2-4dee-bc3a-8e0000a09adf" />


The loss trend quickly levels off. Eventually, the validation MSE reaches **0.018**—a notably low value for this dataset.



### **3.2 Prediction Accuracy**

<img width="800" height="800" alt="Image" src="https://github.com/user-attachments/assets/af67cb69-ed9e-40d1-a7bc-de95161075fa" />

The model reached nearly an **R² of 0.89** during testing, meaning it accounts for about 89% of price variations.



### **3.3 Distribution Comparison**
The blue training set matches closely with red test results in their peak patterns, suggesting the system captured core trends effectively rather than just memorizing cases.



---

## 💡 Part 4: Discussion and Analysis

### **4.1 Challenges Faced**
* **Overfitting:** Initially, training error dropped to <0.005 while validation stayed flat. We fixed this using **Batch Normalization** and **Dropout**.
* **Target Skewness:** Raw prices ($30k to $750k) caused gradient explosions. The **log-transformation** $log(1+y)$ turned the optimization landscape more regular.
* **High-end Homes:** The model initially struggled with homes above $500k. Expanding the first layer to **512 neurons** allowed the network to detect finer patterns.

### **4.2 Result Interpretation**
The tight match in the loss curve confirms the system handles underfitting and overfitting correctly. Points hugging the diagonal line ($y=x$) show the guesses aren’t leaning too high or low.

### **4.3 Model Limitations**
* **Interpretability:** Unlike decision trees, it is harder to pin down exactly how much a specific feature (like a fireplace) adds to the value.
* **Data Scarcity:** Performance on premium houses is highly dependent on the number of high-end examples in the training set.

---

## 📁 Project Resources
* **Document:** [Download Report](./House_Price_Prediction_Report.docx)
* **Dataset:** [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

<p align="center">Made with ❤️ for Data Science</p>
