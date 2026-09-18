# DeepCSAT: E-Commerce Customer Satisfaction Score Prediction

##  Overview
DeepCSAT is an end-to-end predictive deep learning system designed to forecast Customer Satisfaction (CSAT) scores (1–5) in real time using customer support interaction metadata, operational metrics, and text feedback.

##  Key Features
- **Data Cleaning & Imputation**: Preprocessing of 85,907 customer interactions, handling high-sparsity features (`connected_handling_time`), and imputing missing categorical/numerical values.
- **Feature Engineering & NLP**: Extracted ticket resolution times, cyclical date-time signals, and TF-IDF representations from customer feedback remarks.
- **Addressing Class Imbalance**: Evaluated SMOTE and class-weight balancing strategies to address minority dissatisfaction classes (CSAT 1, 2, and 3).
- **Deep Learning Model**: 6-layer Deep Artificial Neural Network (ANN) built with TensorFlow/Keras, utilizing Batch Normalization, Dropout, and L2 regularization.
- **Validation**: 3-Fold Stratified Cross-Validation with dynamic learning rate scheduling and early stopping.
- **Deployment**: Interactive Streamlit web application for real-time inference and score prediction.

##  Installation & Running the App
```bash
# Clone the repository
git clone [https://github.com/abhik99/DeepCSAT-E-Commerce-Customer-Satisfaction-Score-Prediction.git](https://github.com/abhik99/DeepCSAT-E-Commerce-Customer-Satisfaction-Score-Prediction.git)
cd DeepCSAT-E-Commerce-Customer-Satisfaction-Score-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit Application
streamlit run app/app.py
