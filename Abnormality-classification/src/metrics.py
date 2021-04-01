import numpy as np
from sklearn.metrics import confusion_matrix

def update_confusionMatrix(labels,yhat,current_cm):

    tn, fp, fn, tp = confusion_matrix(labels.view(-1).cpu(), yhat.argmax(-1).view(-1).cpu()).ravel()
    running_tn, running_fp, running_fn, running_tp = current_cm#[0],current_cm[1],current_cm[2],current_cm[3]
    running_tn+=tn
    running_fp+=fp
    running_fn+=fn
    running_tp+=tp
    return running_tn, running_fp, running_fn, running_tp

def scores_from_confusionMatrix(tn, fp, fn, tp):
    sn = tp.astype(np.float)/(tp.astype(np.float)+fn.astype(np.float))
    sp = tn.astype(np.float)/(tn.astype(np.float)+fp.astype(np.float))
    ppv= tp.astype(np.float)/(tp.astype(np.float)+fp.astype(np.float))
    f1score = 2*(ppv*sn)/(ppv+sn)
    return sn, sp, ppv, f1score
