# Coronary Heart Disease Prediction Using Machine Learning and Deep Learning  

## 📌 Overview  
Coronary Heart Disease (CHD) is one of the leading causes of global mortality. Early prediction and diagnosis can significantly improve patient outcomes and reduce healthcare burdens.  
This project presents a comparative study between **traditional Machine Learning (ML) models** and a **Deep Neural Network (DNN)** for CHD prediction, using the publicly available **[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)** from Kaggle.  

Our experiments show that while K-Nearest Neighbors (KNN) and DNN both achieve **89.44% accuracy**, the DNN demonstrates superior performance in terms of **AUC (0.939)** and generalization, highlighting its effectiveness for healthcare analytics.  

---

## 🚀 Features  
- Implementation of multiple **ML algorithms**:  
  - Logistic Regression  
  - Decision Tree  
  - Gaussian Naive Bayes  
  - K-Nearest Neighbors (KNN)  

- Development of a **Deep Neural Network (DNN)** with:  
  - Multiple hidden layers  
  - ReLU activation  
  - Dropout for regularization  
  - Softmax output layer  

- **Data preprocessing**: handling missing values, outlier detection, normalization, and one-hot encoding.  
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score, Specificity, Confusion Matrix, and AUC.  
- **Exploratory Data Analysis (EDA)** including correlation heatmaps and feature importance.  
- **Visualization** of training/testing accuracy and ROC curves.  

---

## 📂 Dataset  
- **Source**: [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
- **Size**: 918 records, 12 features  
- **Key Attributes**: Age, Cholesterol, Resting Blood Pressure, ECG Results, Chest Pain Type, Max Heart Rate, ST Slope, etc.  

---

## 🛠️ Tech Stack  
- **Programming Language**: Python  
- **Libraries**:  
  - Pandas, NumPy → Data handling  
  - Matplotlib, Seaborn → Visualization  
  - Scikit-learn → ML models & metrics  
  - TensorFlow, Keras → Deep Learning  

- **Environment**: Google Colab (GPU runtime)  

---

## 📊 Results  

| Model               | Accuracy | AUC   | Precision | Recall | F1-Score |  
|---------------------|----------|-------|-----------|--------|----------|  
| **Deep Neural Net** | **0.894**| **0.939** | 0.89      | 0.89   | 0.89     |  
| Logistic Regression | 0.882    | 0.932 | 0.88      | 0.88   | 0.88     |  
| Gaussian NB         | 0.873    | 0.913 | 0.87      | 0.87   | 0.87     |  
| Decision Tree       | 0.798    | 0.798 | 0.80      | 0.80   | 0.80     |  
| KNN                 | **0.894**| 0.928 | 0.89      | 0.89   | 0.89     |  

📌 **Best Overall Model** → **DNN** (highest AUC & generalization ability)  

---

## 📈 Visualizations  
- ROC Curve comparisons  
- Correlation heatmap of key features  
- Accuracy and loss curves across epochs  
- Feature importance from ML models  

---

## 🔮 Future Scope  
- Use real-world hospital datasets including lifestyle and family history factors.  
- Integration with **Electronic Health Records (EHRs)** and wearable devices for real-time risk prediction.  
- Incorporate **Explainable AI (XAI)** for better interpretability.  
- Explore **Federated Learning** for privacy-preserving model training.  

---

## 👨‍💻 Author  
**Saurabh Singh Gurjar**  
Student, Madhav Institute of Technology and Science  
📧 [saurabhsinghgurjar975@gmail.com](mailto:saurabhsinghgurjar975@gmail.com)  

---

## 📜 References  
- [Kaggle Heart Failure Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
- Schmidhuber, J. *Deep Learning in Neural Networks: An Overview*. Neural Networks, 2015.  
- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning*. Springer, 2009.  
- Mandadi et al., ICCMC 2023 – *Machine Learning and Deep Learning Techniques on Risk Prediction of CHD*  
- Miao & Miao, IJACSA 2018 – *Coronary Heart Disease Diagnosis using Deep Neural Networks*  

