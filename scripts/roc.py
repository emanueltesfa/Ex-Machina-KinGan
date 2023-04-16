import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
     # used to Plot ROC curves
    scores_raw = pd.read_csv("scores_100.csv").to_numpy()[1:,1:]
    fake_scores_raw = pd.read_csv("fake_scores_100.csv").to_numpy()[1:,1:]
    scores = scores_raw[:,:3]
    print(scores.shape,scores_raw.shape)
    scores[:,0] = np.mean(scores_raw[:,0:2],axis=1)
    scores[:,1] = np.mean(scores_raw[:,2:4],axis=1)
    scores[:,2] = scores_raw[:,4]
    fake_scores = fake_scores_raw[:,:3]
    fake_scores[:,0] = np.mean(fake_scores_raw[:,0:2],axis=1)
    fake_scores[:,1] = np.mean(fake_scores_raw[:,2:4],axis=1)
    fake_scores[:,2] = fake_scores_raw[:,4]

    labels = ["Average Parent to Real","Average Parent to Generated","Real to Generated"]
    lines = []
    for i in range(3):
        print(scores.shape)
        pairwise = scores[:,i].reshape(-1)
        f_pairwise = fake_scores[:,i].reshape(-1)
        print(pairwise.shape, pairwise.max(),pairwise.min())
        print(np.ones_like(scores).shape)
        true = np.ones_like(pairwise)
        f_true = np.zeros_like(f_pairwise)
        preds = np.concatenate([pairwise,f_pairwise])
        full_true = np.concatenate([true,f_true])
        print(preds.shape)
        print(full_true.shape)
        fpr, tpr, thresholds = metrics.roc_curve(full_true,  preds)
        
        print(thresholds)
        #create ROC curve
        line, = plt.plot(fpr,tpr, label=f"{labels[i]}: {np.round(metrics.auc(fpr,  tpr),2)}")
        lines.append(line)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(handles=lines)
    plt.title("AUC Curves for Potential Losses")
    plt.show()