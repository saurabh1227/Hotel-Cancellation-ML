# Hotel Booking Cancellation Prediction

## a. Problem Statement
Predict whether a hotel booking will be canceled (`is_canceled` = 1) or not (0) using booking details such as lead time, deposit type, market segment and special requests. Five classification models are implemented and compared on the same dataset using Accuracy, AUC, Precision, Recall, F1 and MCC. The trained models are deployed in a Streamlit app.

## b. Dataset Description
- Source: Kaggle, Hotel Booking dataset [PASTE KAGGLE LINK HERE]
- Type: binary classification, target = `is_canceled`
- Original size: [ROWS] rows x [COLUMNS] columns. After cleaning and encoding, a stratified sample of 25,000 rows with [NUMBER] input features was used (80/20 train-test split, stratified).
- Preprocessing: removed leakage columns (`reservation_status`, `reservation_status_date`, `assigned_room_type`), dropped irrelevant / high-cardinality columns (`arrival_date_year`, `country`, `agent`, `company`), filled missing `children` with 0, replaced negative `adr` with the median, removed rows with 0 adults and outlier rows (10 children / 10 babies), label-encoded the month, and one-hot encoded the remaining categorical columns.

## c. GitHub Repository Link
[PASTE GITHUB REPO LINK HERE]

Live app: [PASTE STREAMLIT APP LINK HERE]

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8046 | 0.8459 | 0.8227 | 0.6030 | 0.6959 | 0.5715 |
| Decision Tree | 0.8082 | 0.8643 | 0.7908 | 0.6564 | 0.7174 | 0.5798 |
| KNN | 0.7874 | 0.8337 | 0.7381 | 0.6613 | 0.6976 | 0.5363 |
| Naive Bayes | 0.5574 | 0.7819 | 0.4516 | 0.9024 | 0.6019 | 0.2828 |
| Support Vector Machine | 0.8066 | 0.8576 | 0.8332 | 0.5982 | 0.6964 | 0.5766 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good precision (0.82) but low recall (0.60); it misses many real cancellations. A solid, simple baseline. |
| Decision Tree | Best overall: highest AUC, F1 and MCC, with a good balance of precision and recall. |
| KNN | Slightly weaker than the others; sensitive to feature scaling and to the many one-hot columns. |
| Naive Bayes | Very high recall (0.90) but very low precision and accuracy; the Gaussian and independence assumptions do not fit one-hot and skewed features. |
| Support Vector Machine | Highest precision (0.83) and second-best AUC, but recall is low (0.60), so it is conservative about predicting cancellations. |
| **Overall, Winner for your dataset?** | **Decision Tree.** It has the highest AUC (0.8643), Accuracy (0.8082), F1 (0.7174) and MCC (0.5798), with the best balance between precision (0.7908) and recall (0.6564). Support Vector Machine is a close second (AUC 0.8576, highest precision 0.8332), while Naive Bayes is the weakest (Accuracy 0.5574, MCC 0.2828). |
