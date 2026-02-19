X_noise = X_test.copy()
X_noise['temperature'] += np.random.normal(0, 2, X_noise.shape[0])           # noise
X_noise['temperature'] += np.linspace(0, 10, X_noise.shape[0])              # drift

pred_noise = model.predict(X_noise)
from sklearn.metrics import classification_report
print("Performance on noisy/drifted data:")
print(classification_report(y_test, pred_noise))
