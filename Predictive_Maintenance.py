
#hobbyprojekt

import pandas as pd

df_maintenance = pd.read_csv("predictive_maintenance.csv")


#från scikit-learn-biblioteket hämtar jag train_test_split och RandomForestClassifier som kan klassificera y till 0 eller 1.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



X = df_maintenance.drop(columns=["Target", "Failure Type", "UDI", "Product ID"])
y = df_maintenance["Target"]


X = pd.get_dummies(X, columns=["Type"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix

print("Classification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
