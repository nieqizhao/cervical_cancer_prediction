import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import shap


df_filter = pd.read_csv("df_final_file.csv")
SEED = 22
y = df_filter['Cervical Cancer Status']
X = df_filter.drop(['Cervical Cancer Status'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=SEED) 
sm = SMOTE(random_state=SEED)
X_train_scaled, y_train_scaled = sm.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=SEED)
model.fit(X_train_scaled, y_train_scaled)  
y_score = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))
# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cbar = False, fmt = "d", linewidths = .5, cmap = "Blues", ax = ax1,annot_kws={'fontsize': 12})
ax1.tick_params(labelsize=14)
ax1.set_title("Confusion Matrix",fontsize=20)
ax1.set_xlabel("Predicted class",fontsize=20)
ax1.set_ylabel("Actual class",fontsize=20)
plt.savefig('cm_rf.pdf', format='pdf', bbox_inches='tight')

#shap
explainer = shap.TreeExplainer(model, X_train_scaled, feature_dependence="independent")
shap_values = explainer.shap_values(X_test)

plt.style.use('ggplot')
sns.set_style('white')
plt.figure(figsize=(20,20), dpi=300)
plt.rcParams['font.sans-serif'] = ['Times New Roman']  
plt.rcParams['axes.unicode_minus'] = False  
shap.summary_plot(shap_values[1], X_test, feature_names=X.columns,show=False, plot_type="dot")
plt.savefig('summary_plot_1.pdf', format='pdf', bbox_inches='tight')
plt.show()


labels_mapping = {0: 'No',1: 'Yes'}
plt.style.use('ggplot')
sns.set_style('white')
plt.figure(figsize=(20,20), dpi=300)
plt.rcParams['font.sans-serif'] = ['Times New Roman']   
shap.summary_plot(shap_values, X_test, feature_names=X.columns,show=False,class_names=labels_mapping,class_inds = "original")
plt.savefig('summary_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()



feature_main = 'SCC'         
feature_interaction = 'HPV 16 or HPV18' 

shap.dependence_plot(
    feature_main,
    shap_values[1],       
    X_test,
    feature_names=X_test.columns,
    dot_size=5,
    interaction_index=feature_interaction,
    show=False
)

plt.ylabel('SHAP value', fontsize=10, labelpad=10)
plt.savefig(f'./shap_interaction_{feature_main}_{feature_interaction}.pdf', bbox_inches='tight')
plt.close()


for index in X_test.columns:
    shap.dependence_plot(index, shap_values[1], X_test, feature_names=X_test.columns, dot_size=2, interaction_index=None,show=False)
    plt.ylabel('SHAP value', fontsize=10, labelpad=10)
    plt.savefig(r'./shap_single_%s.pdf'% (index), bbox_inches='tight')
    

ind = 0
shap.force_plot(explainer.expected_value[1], shap_values[1][ind], round(X_test.iloc[ind,:],3),feature_names=X.columns,matplotlib=True,show=False)
plt.savefig('71_0.pdf', dpi=300, bbox_inches='tight')
plt.show()

ind = 2
shap.force_plot(explainer.expected_value[1], shap_values[1][ind], round(X_test.iloc[ind,:],3),feature_names=X.columns,matplotlib=True,show=False)
plt.savefig('2_0.pdf', dpi=300, bbox_inches='tight')
plt.show()

