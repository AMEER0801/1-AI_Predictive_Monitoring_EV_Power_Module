import pandas as pd
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feat_imp)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.show()
