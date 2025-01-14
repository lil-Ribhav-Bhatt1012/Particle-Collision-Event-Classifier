# **High-Energy Particle Classification using Machine Learning**

## 📋 **Overview**
This project involves building a machine learning model to classify events from high-energy particle collisions into two categories:
- **Signal (`s`)**: Events of scientific interest.
- **Background (`b`)**: Other non-relevant events.

The dataset consists of physics-derived features, and a **Random Forest Classifier** was implemented to predict the event type. The project includes insights into feature importance, class distribution, and evaluation metrics.

---

## 🛠️ **Technologies and Tools**
- **Programming Language**: Python
- **Key Libraries**:
  - **Data Manipulation**: Pandas, NumPy
  - **Machine Learning**: Scikit-learn
  - **Visualization**: Matplotlib
  - **Data Imbalance Handling**: Imbalanced-learn (optional)

---

## 📂 **Repository Structure**
```
├── notebooks/
│   └── particle_classification.ipynb   # Jupyter Notebook with full analysis
├── particle_classification.py          # Python script for end-to-end execution
├── README.md                           # Project documentation
├── requirements.txt                    # Dependencies for replication
└── submission.csv                      # Final submission file for predictions
```

---

## 🚀 **Installation**
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧑‍💻 **Usage**
### **Running the Notebook**
1. Open the notebook:
   ```bash
   jupyter notebook notebooks/particle_classification.ipynb
   ```
2. Follow the structured cells to preprocess data, train the model, evaluate performance, and generate predictions.

### **Running the Python Script**
1. Execute the script:
   ```bash
   python particle_classification.py
   ```
2. The `submission.csv` file will be generated in the repository's root directory.

---

## 📊 **Project Details**
### **Dataset Overview**
1. **Training Dataset**:
   - 250,000 events labeled as `s` (signal) or `b` (background).
   - 30 numerical features derived from physical experiments.
2. **Test Dataset**:
   - 550,000 unlabeled events requiring classification.

### **Model**
- **Algorithm**: Random Forest Classifier
- **Metrics**:
  - **Accuracy**: `85.4%`
  - **ROC AUC Score**: `0.91`

### **Key Analysis**:
1. **Feature Importance**:
   - Identified and visualized the top 10 most impactful features for classification.
2. **Class Distribution**:
   - Highlighted the imbalance with ~70% background (`b`) and ~30% signal (`s`) in predictions.

---

## 📁 **Submission File**
The `submission.csv` file contains predictions for the test dataset, structured as:
- **EventId**: Unique identifier for each event.
- **RankOrder**: Rank of events by signal likelihood.
- **Class**: Predicted label (`s` or `b`).

### Example:
| EventId | RankOrder | Class |
|---------|-----------|-------|
| 350000  | 1         | b     |
| 350001  | 202767    | b     |
| 350002  | 382001    | b     |
| 350003  | 522335    | s     |

---

## ⚡ **How to Reproduce**
1. Prepare the datasets (`training.csv` and `test.csv`) in the repository's root directory.
2. Run the notebook or script to preprocess data, train the model, and generate the submission file.

---

## 🔍 **Future Enhancements**
1. Address class imbalance using:
   - **SMOTE**: Synthetic Minority Over-sampling Technique.
   - **Class Weighting**: Adjust model weights to balance `s` and `b`.
2. Experiment with advanced models like **XGBoost** and **LightGBM** for improved performance.
3. Optimize hyperparameters with **GridSearchCV** for better predictions.

---

## 🙏 **Acknowledgments**
This project draws inspiration from challenges in high-energy physics, applying machine learning to advance research in particle collision analysis.

---

## 📦 **Requirements**
Below is the `requirements.txt` content:

```text
pandas
numpy
scikit-learn
matplotlib
jupyter
imbalanced-learn
```
```
