#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for evaluation metrics
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    classification_report, roc_curve, log_loss, auc, 
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix, precision_recall_fscore_support
)

import warnings
warnings.filterwarnings('ignore')


# ## For Regression

# In[7]:


def regression_model_summary(y_true, y_pred):
    r_squared = round(r2_score(y_true, y_pred), 3)
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    print("The r-squared of the model is {} and its mean absolute error is {}.".format(r_squared, mae))
    
    res = {'R2':r_squared, 'RMSE':rmse, 'MAE':mae}
    return res


# ## For Classification

# In[9]:


def classification_eval_metrics(y_true, y_pred, y_score):
    (pr, rc, f1, _) = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    logloss = log_loss(y_true, y_pred)
        
    res = {'Accuracy': acc, 'F1-Score':f1, 'Recall':rc, 
           'Precision':pr, 'ROC-AUC':roc_auc, 'Log-Loss':logloss}
    formatted_res = {key: round(value, 3) for key, value in res.items()}
    
    # Creating a new dictionary with formatted values
    print(classification_report(y_true, y_pred, target_names=['No', 'Yes']))
    
    return formatted_res


# In[7]:


def classification_model_summary(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
    
    print(f"The accuracy of the model is {acc:.3f} and its F1 score is {f1:.3f}.")


# In[3]:


def plot_roc(y_true, y_score, classifier_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label = 1)
    
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--' )
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for ' + classifier_name)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.show()


# In[4]:


def plot_all_roc(roc_data):
    plt.figure(figsize=(8,6))
    
    for x, y in roc_data.items():  
        classifier_name = x
        (fpr, tpr, auc_val) = y
        plt.plot(fpr, tpr, label=f'{classifier_name} (AUC = {auc_val:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves')
    plt.legend(loc='lower right', fontsize=8)
    plt.show()


# In[5]:


def roc_auc_vals(y_true, y_score):
    fpr, tpr, _ =roc_curve(y_true, y_score, pos_label = 1)
    auc_val=auc(fpr, tpr)  
    
    return (fpr, tpr, auc_val)


# In[6]:


def plot_cm(y_true, y_pred):
    cm=confusion_matrix(y_true, y_pred)
    cm_matrix = pd.DataFrame(data=cm, columns=['Predicted Negative:0', 'Predicted Positive:1'], 
                                 index=['Actual Negative:0', 'Actual Positive:1']) 
    
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

