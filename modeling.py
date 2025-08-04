
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,roc_curve, auc,cohen_kappa_score,confusion_matrix,precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def two_class_evaluation(y_test, y_pred, y_prob):
    conf = confusion_matrix(y_test, y_pred)
    acc = (conf[0, 0] + conf[1, 1]) / (conf[0, 0] + conf[0, 1] + conf[1, 0] + conf[1, 1])
    TPR = (conf[1, 1]) / (conf[1, 0] + conf[1, 1])
    FPR = (conf[0, 1]) / (conf[0, 0] + conf[0, 1])
    TNR = (conf[0, 0]) / (conf[0, 0] + conf[0, 1])
    FNR = (conf[1, 0]) / (conf[1, 0] + conf[1, 1])
    if conf[0, 1] + conf[1, 1] == 0:
        PPV = 0
    else:
        PPV = (conf[1, 1]) / (conf[0, 1] + conf[1, 1])
    if conf[0, 0] + conf[1, 0] == 0:
        NPV = 0
    else:
        NPV = (conf[0, 0]) / (conf[0, 0] + conf[1, 0])
    if PPV + TPR == 0:
        F1 = 0
    else:
        F1 = (2 * PPV * TPR) / (PPV + TPR)
    fpr1, tpr1, threshold1 = roc_curve(y_test, y_prob)
    AUROC = auc(fpr1, tpr1)
    precision2, recall2, thresholds2 = precision_recall_curve(y_test, y_prob)
    AUPRC = auc(recall2, precision2)
    kappa = cohen_kappa_score(y_test,y_pred)
    scores_dict = {
        'accuracy' : round(acc, 3), 'Sensitivity' : round(TPR, 3),
        'Specificity' : round(TNR, 3), 'Precision' :  round(PPV, 3),
        'F1-Score' : round(F1, 3), 'AUROC' : round(AUROC, 3),
        'AUPRC' : round(AUPRC, 3),'kappa': round(kappa, 3),
    }
    return scores_dict


df_filter = pd.read_csv("df_final_file.csv")
SEED = 22
y = df_filter['Cervical Cancer Status']
X = df_filter.drop(['Cervical Cancer Status'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=SEED) 
sm = SMOTE(random_state=SEED)
X_train_scaled, y_train_scaled = sm.fit_resample(X_train, y_train)


#Grid Search for hyperparameters 
models_params = {
    "SVM": (
        SVC(probability=True),
        {
            "C": [0.01, 0.1, 1, 10],
            "gamma": ["scale", "auto"],
        },
    ),
    "LR": (
        LogisticRegression(max_iter=10000),
        {
            "C": [0.01, 0.1, 1, 10],
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
        },
    ),
    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5, 7],
        },
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=SEED),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.05, 0.1, 1.0],
        },
    ),
    "RF": (
        RandomForestClassifier(random_state=SEED),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 3, 5,10],
            "max_features": ["sqrt", "log2"],
        },
    ),
    "DT": (
        DecisionTreeClassifier(random_state=SEED),
        {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
    ),
    "LGB": (
        LGBMClassifier(random_state=SEED),
        {
            "num_leaves": [31, 50, 100],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.01, 0.1],
        },
    ),
    "MLP": (
        MLPClassifier(random_state=SEED),
        {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "alpha": [0.0001, 0.001],
        },
    ),
    
}

def eval_on_val(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]  
    return {
        "ACC": accuracy_score(y_val, y_pred),
        "F1": f1_score(y_val, y_pred),
        "AUROC": roc_auc_score(y_val, y_prob),
        "AUPRC": average_precision_score(y_val, y_prob),
    }

results = []

for name, (base_clf, param_grid) in models_params.items():
    best_metrics = {}
    best_params = {}
    best_score = -np.inf  
    print(f"=== Tuning {name} ===")

    grid = (
        [dict()] if not param_grid else
        (dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values()))
    )

    for params in grid:
        pipe = Pipeline(
            steps=[
                ("clf", base_clf.set_params(**params)),
            ]
        )
        pipe.fit(X_train_scaled, y_train_scaled)
        metrics_val = eval_on_val(pipe, X_test, y_test)

        if metrics_val["AUROC"] > best_score:
            best_score = metrics_val["AUROC"]
            best_metrics = metrics_val
            best_params = params

    results.append(
        {
            "Model": name,
            "Best Params": best_params,
            **best_metrics,
        }
    )


results_df = pd.DataFrame(results).sort_values(by="AUROC", ascending=False)
print(results_df)
results_df.to_csv("model_hyperparameter_summary_noCV.csv", index=False)



model = CatBoostClassifier(random_state=SEED, verbose=0)
model.fit(X_train_scaled, y_train_scaled)

y_score = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
scores_dict = two_class_evaluation(y_test, y_pred, y_score)
print(scores_dict)

for lr in [ 0.05, 0.1]:
    for depth in [4, 6, 8]:
        model = CatBoostClassifier(learning_rate=lr,depth=depth,random_state=SEED, verbose=0)
        model.fit(X_train_scaled, y_train_scaled)
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        scores_dict = two_class_evaluation(y_test, y_pred, y_score)
        print('max_depth is '+ str(depth))
        print('learning_rate is '+ str(lr))
        print(scores_dict)
        

#ROC curve
theme = plt.get_cmap('tab20')
plt.style.use('ggplot')
sns.set_style('white')
colors = [theme(i) for i in range(11)]

plt.figure(figsize=(10, 10), dpi=300, facecolor='white')
List2 = ['SVM', 'LR', 'KNN', 'XGBoost', 'AdaBoost', 'RF', 'DT', 'NB', 'LGB', 'CatBoost', 'MLP']
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 使用 Times New Roman 字体

for index in range(11):
    if index == 0:
        model = SVC(C=10, gamma='scale', probability=True)
    elif index == 1:
        model = LogisticRegression(C=0.1, max_iter=10000)
    elif index == 2:
        model = KNeighborsClassifier(n_neighbors=7, weights='uniform')
    elif index == 3:
        model = XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=3)
    elif index == 4:
        model = AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=SEED)
    elif index == 5:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    elif index == 6:
        model = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=SEED)
    elif index == 7:
        model = GaussianNB()
    elif index == 8:
        model = LGBMClassifier(num_leaves=31, max_depth=-1, learning_rate=0.1, random_state=SEED)
    elif index == 9:
        model = CatBoostClassifier(learning_rate=0.05, depth=6, random_state=SEED, verbose=0)
    elif index == 10:
        model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10000, random_state=SEED, alpha=0.0001)

    model.fit(X_train_scaled, y_train_scaled)
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 计算最佳阈值
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = threshold[ix]
    print(f"{List2[index]}: Best Threshold = {best_thresh:.3f}, AUROC = {roc_auc:.3f}")

    plt.plot(fpr, tpr, linewidth=2, color=colors[index],
             label=f"{List2[index]} (AUROC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
plt.xlabel('FPR', fontsize=20)
plt.ylabel('TPR', fontsize=20)
plt.title('ROC curve', fontsize=25)
plt.legend(loc="lower right", fontsize=10)
plt.savefig('AUROC.pdf', format='pdf', bbox_inches='tight')
plt.show()


#PR curve
plt.style.use('ggplot')
sns.set_style('white')
theme = plt.get_cmap('tab20')
colors = [theme(i) for i in range(11)]


plt.figure(figsize=(10, 10),dpi=300,facecolor='white')
List2 = ['SVM','LR', 'KNN','XGBoost', 'AdaBoost', 'RF','DT', 'NB','LGB','CatBoost','MLP']
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 黑体
for index in [0,1,2,3,4,5,6,7,8,9,10]:
    if index == 0:
        model = SVC(C= 10, gamma= 'scale',probability=True)
    elif index == 1:
        model = LogisticRegression(C=0.1,max_iter=10000)
    elif index == 2:
        model = KNeighborsClassifier(n_neighbors= 7, weights= 'uniform')
    elif index == 3:
        model = XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth= 3)
    elif index == 4:
        model = AdaBoostClassifier(n_estimators=100, learning_rate=0.05,random_state=SEED)
    elif index == 5:
        model = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=SEED)
    elif index == 6:
        model = DecisionTreeClassifier(max_depth= 10, min_samples_split=10,random_state=SEED)
    elif index == 7:
        model = GaussianNB()
    elif index == 8:
        model = LGBMClassifier(num_leaves= 31, max_depth= -1, learning_rate=0.1,random_state=SEED)
    elif index == 9:
        model = CatBoostClassifier(learning_rate=0.05,depth=6,random_state=SEED, verbose=0)
    elif index == 10:
        model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10000, random_state=SEED,alpha=0.0001)
        
    if index in [0,1,2,3,4,5,6,7,8,9,10]:
        model.fit(X_train_scaled, y_train_scaled)
        y_score = model.predict_proba(X_test)[:,1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        prc_auc = auc(recall,precision)
        plt.plot(recall, precision,linewidth=2,color=colors[index],label = List2[index]+" (AUPRC="+str(round(prc_auc, 3))+")")  

plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
plt.xlabel('Recall',fontsize=20)
plt.ylabel('Precision',fontsize=20)
plt.title('Precision recall Curve',fontsize=25)
plt.legend(loc="lower right",fontsize=12)
plt.savefig('PR_curve.pdf', format='pdf', bbox_inches='tight')
plt.show()


#evaluation 
def roc_auc_ci(y_true, y_score, n_bootstraps=1000, alpha=0.95, seed=42):
    """
    Calculate AUROC and its confidence interval using bootstrap.
    """
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        
        if len(np.unique(y_true.iloc[indices])) < 2:
           
            continue
        score = roc_auc_score(y_true.iloc[indices], y_score[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    mean_score = np.mean(sorted_scores)
    return round(mean_score, 3), round(lower, 3), round(upper, 3)


def two_class_evaluation1(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    mean_auroc, lower_auroc, upper_auroc = roc_auc_ci(y_true, y_score)
    auroc_str = f"{mean_auroc} ({lower_auroc}, {upper_auroc})"

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    sorted_indices = np.argsort(recall_curve)
    auprc = np.trapz(precision_curve[sorted_indices], recall_curve[sorted_indices])

    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        'accuracy': round(accuracy, 3),
        'Sensitivity': round(sensitivity, 3),
        'Specificity': round(specificity, 3),
        'Precision': round(precision, 3),
        'F1-Score': round(f1, 3),
        'AUROC (95% CI)': auroc_str,
        'AUPRC': round(auprc, 3),
        'kappa': round(kappa, 3)
    }



model_list = ['NB','KNN','SVM','LR','DT','AdaBoost','XGBoost','RF','LGB','CatBoost','MLP']
results = []


for model_item in model_list:
    if model_item == 'NB':
        model = GaussianNB()
    elif model_item == 'KNN':    
        model = KNeighborsClassifier(n_neighbors=7)
    elif model_item == 'SVM':
        model = SVC(probability=True)
    elif model_item == 'LR':
        model = LogisticRegression(max_iter=10000)
    elif model_item == 'DT':    
        model = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=SEED)
    elif model_item == 'AdaBoost':    
        model = AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=SEED)
    elif model_item == 'XGBoost':    
        model = XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=3)
    elif model_item == 'RF':    
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    elif model_item == 'LGB':    
        model = LGBMClassifier(num_leaves=31, max_depth=-1, learning_rate=0.1, random_state=SEED)
    elif model_item == 'CatBoost':    
        model = CatBoostClassifier(learning_rate=0.05, depth=6, random_state=SEED, verbose=0)
    elif model_item == 'MLP':    
        model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10000, random_state=SEED)

    model.fit(X_train_scaled, y_train_scaled)

    y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    J = tpr - fpr
    best_thresh = thresholds[np.argmax(J)]
    y_pred = (y_score >= best_thresh).astype(int)

    scores = two_class_evaluation1(y_test, y_pred, y_score)
    scores['model'] = model_item
    scores['best_thresh'] = round(best_thresh, 3)

    results.append(scores)

df_results = pd.DataFrame(results)
df_results = df_results[['model', 'accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUROC (95% CI)', 'AUPRC', 'kappa', 'best_thresh']]

print(df_results)

df_results.to_csv("model_comparison_best_threshold.csv", index=False)

