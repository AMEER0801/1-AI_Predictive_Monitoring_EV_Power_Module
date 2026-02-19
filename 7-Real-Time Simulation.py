risk_scores = model.predict_proba(X_test)[:,1]

plt.figure(figsize=(10,5))
plt.plot(risk_scores[:100])
plt.title("Real-Time Risk Score Simulation")
plt.xlabel("Time Step")
plt.ylabel("Failure Risk Probability")
plt.show()
