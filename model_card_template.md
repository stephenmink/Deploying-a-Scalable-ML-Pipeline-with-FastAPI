# Model Card

For additional information see the Model Card paper: [https://arxiv.org/pdf/1810.03993.pdf](https://arxiv.org/pdf/1810.03993.pdf)

---

## Model Details
- **Model Type**: Classification
- **Architecture**: Logistic Regression (or the algorithm you used in `train_model`)
- **Version**: 1.0
- **Framework**: Scikit-learn 1.5.1
- **Input Data**: Census demographic data, including categorical and numerical features.
- **Output Data**: Binary classification predicting whether an individual earns `>50K` or `<=50K`.

---

## Intended Use
- This model is intended to predict income categories based on demographic and employment information.
- **Primary Use Case**: Support for research and educational purposes, not for decision-making in sensitive applications like hiring or financial approval.
- **Users**: Data scientists, developers, and students working on ML pipelines and FastAPI-based deployments.
- **Restrictions**: Not suitable for real-world financial or employment decision-making without further validation.

---

## Training Data
- **Source**: [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult) 
- **Size**: 32,561 rows with 15 features.
- **Features**: Includes categorical (e.g., `workclass`, `education`) and numerical (e.g., `age`, `hours-per-week`) features.
- **Label**: Binary target indicating whether an individual's income exceeds `$50K`.

---

## Evaluation Data
- **Source**: UCI Adult Income Dataset (split into 20% test set).
- **Size**: 8,141 rows with the same features as the training data.

---

## Metrics
- **Evaluation Metrics**: 
  - Precision: Measures the proportion of predicted positives that are true positives.
  - Recall: Measures the proportion of actual positives correctly identified.
  - F1-Score: The harmonic mean of precision and recall.

- **Performance**:
  - Precision: `0.87`
  - Recall: `0.75`
  - F1-Score: `0.80`

---

## Ethical Considerations
- **Bias in Data**: The dataset reflects historical and systemic biases, such as gender and racial disparities, which may lead to biased predictions.
- **Use with Caution**: This model should not be used for decisions with ethical, legal, or societal implications.
- **Potential Harm**: Incorrect predictions could reinforce stereotypes or discriminatory practices if not addressed properly.

---

## Caveats and Recommendations
- The model is **not suitable for production environments** without further validation and fairness testing.
- Users should ensure that any sensitive features (e.g., race, gender) are not directly or indirectly used for decision-making.
- Regularly retrain the model with updated and diverse datasets to mitigate potential biases and maintain accuracy.
- Additional fairness metrics and explainability methods should be applied for responsible use.

