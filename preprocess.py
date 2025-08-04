import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Lasso,LassoCV
from sklearn.impute import SimpleImputer
from IPython.core.pylabtools import figsize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_curve, auc, precision_recall_curve



def count_missing_values(dataframe):
    
    total = dataframe.isnull().sum()
    percent = (dataframe.isnull().sum())*100/(len(dataframe))
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data = missing_data[missing_data['Total']>0]
    missing_data.sort_values('Total', ascending=False, inplace=True)
    print(missing_data)
    

def mean(df,label):
    df.iloc[:,:] = SimpleImputer(strategy='mean').fit_transform(df)
    print('Missing values have been imputed!')
    return df

def min_max(df,label):
    X = df.drop([label],axis=1)
    y = df[label]
    X = X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df = pd.concat([X,y],axis=1)
    print('Min-Max has been normalization!')
    return df 


def optimal_lambda_value(X_train,y_train):
        Lambdas = np.logspace(-5, 2, 200)   
        lasso_cofficients = []
        for Lambda in Lambdas:
            lasso = Lasso(alpha = Lambda, normalize=True, max_iter=20000)
            lasso.fit(X_train,y_train)
            lasso_cofficients.append(lasso.coef_)
        plt.plot(Lambdas, lasso_cofficients)
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('Cofficients')
        plt.show()
        plt.savefig('lasso_feature.pdf', format='pdf', bbox_inches='tight')

        lasso_cv = LassoCV(alphas = Lambdas, normalize=True, cv = 5, max_iter=20000)
        lasso_cv.fit(X_train,y_train)
        lasso_best_alpha = lasso_cv.alpha_
        print(lasso_best_alpha)
        return lasso_best_alpha
    

def lasso_select(X_train,y_train,lasso_best_alpha,number):
    model = Lasso(alpha = lasso_best_alpha, normalize=True, max_iter=10000)
    model.fit(X_train,y_train)
    newcolumns = pd.Series(data=model.coef_, index=X.columns).abs().nlargest(number+1).index
    X_train = X_train[newcolumns]
    print('Based on Lasso model' + str(number+1) + 'features:')
    print(list(X_train.columns))
    return X_train, y_train

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


data_df = pd.read_excel("dataset.xlsx", engine='openpyxl', sheet_name=0)
data_df = data_df.replace(to_replace=r'^\s*$', value=np.nan, regex=True)

#count missing data
count_missing_values(data_df)

#Feature engineering
data_df['BMI'] = data_df['Weight'] / (data_df['Height'] / 100) ** 2 

#imputation
df = mean(data_df,'Cervical Cancer Status')

#normalizing
filled_df = min_max(df,'Cervical Cancer Status')

SEED=22
y = filled_df['Cervical Cancer Status']
X = filled_df.drop(['Cervical Cancer Status'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=SEED)
sm = SMOTE(random_state=SEED)
X_train_scaled, y_train_scaled = sm.fit_resample(X_train, y_train)


sns.set_style("white")
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.sans-serif'] = ['Times New Roman']  
plt.figure(figsize=(10, 10),dpi=300)

#Feature selection
optimal_lambda_value(X_train_scaled,y_train_scaled)

lasso_best_alpha = 0.0017834308769319094
fe_auc =[]
fe_number = []
SEED=22
y = df['Cervical Cancer Status']
X = df.drop(['Cervical Cancer Status'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=SEED) 
for number in list(range(1, 26)):    
    X_train1, y_train1 = lasso_select(X_train,y_train,lasso_best_alpha,(number-1))
    sm = SMOTE(random_state=SEED)
    X_train_scaled, y_train_scaled = sm.fit_resample(X_train1, y_train1)   
    model = RandomForestClassifier(random_state=SEED)
    model.fit(X_train_scaled, y_train_scaled)
    X_test1 = X_test[X_train1.columns]
    y_score = model.predict_proba(X_test1)[:, 1]
    y_pred = model.predict(X_test1)
    scores_dict = two_class_evaluation(y_test, y_pred, y_score)
    print("特征数："+str(number))
    print(scores_dict)
    fe_number.append(number)
    fe_auc.append(scores_dict['AUROC'])
    

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False 
fe_data = pd.DataFrame({'number':fe_number,'auc':fe_auc})
plt.figure(figsize=(10,10), dpi=300)
plt.plot(fe_data['number'],fe_data['auc'],marker='o')
plt.ylim([0.6, 1.01])
plt.xlabel('number of features', fontsize=20)
plt.ylabel('AUROC', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('The number of selection features', fontsize=25)
plt.grid()
plt.tight_layout()
plt.savefig('selection_features_number.pdf', format='pdf', bbox_inches='tight')

df_filter = df[['NILM', 'HSIL', 'Squamous Cell Carcinoma', 'P16-', 'P16+', 'ASC-US', 'HPV 16 or HPV18', 'age', 'CEA', 'SCC', 'CA199','Cervical Cancer Status']]
df_filter.to_csv('df_final_file.csv',index=None)
