# -*- coding: utf-8 -*-
"""
    Evaluation loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
from copy import deepcopy
import metrics
import pdb
def evaluate(model, dataloader, args):

    """ Evaluates a given model and dataset.
    """
    model.eval()
    sample_count = 0
    running_loss = 0
    running_acc = 0
    running_acc_topk = 0
    running_tp, running_fp,running_tn,running_fn = 0,0,0,0

    if args.extract_representation == True:

        all_labels      = np.zeros((len(dataloader)*dataloader.batch_size,1))
        all_predictions = np.zeros((len(dataloader)*dataloader.batch_size,1))
        all_output_activations = np.zeros((len(dataloader)*dataloader.batch_size,2))

    k = 2
    with torch.no_grad():
        for iter, sample in enumerate(dataloader):
            inputs,labels = sample['data'], sample['labels']
            if args.half_precision:
                inputs = inputs.type(torch.HalfTensor).cuda(non_blocking=True)
            else:
                inputs = inputs.type(torch.FloatTensor).cuda(non_blocking=True)

            labels = labels.type(torch.LongTensor).cuda(non_blocking=True)

            if args.extract_representation == True:
                # adapt this code below to each network
                if args.use_net == 'resnet':
                    tmp_model = deepcopy(model)
                    my_model = nn.Sequential(*list(tmp_model.modules()))
                    my_model[0].fc=my_model[0].fc[0]
                    y_intermediate_activations = my_model[0](inputs)

                elif args.use_net == 'squeezenet':
                    tmp_model = deepcopy(model)
                    my_model = nn.Sequential(*list(tmp_model.modules()))
                    my_model[0].classifier = my_model[0].classifier[:2]
                    my_model[0].classifier[1] = my_model[0].classifier[1][0]
                    y_intermediate_activations = my_model[0](inputs)

                elif args.use_net == 'densenet':
                    tmp_model = deepcopy(model)
                    my_model = nn.Sequential(*list(tmp_model.modules()))
                    my_model[0].classifier= my_model[0].classifier[0]
                    y_intermediate_activations = my_model[0](inputs)
                else:
                    print('No net to predict.')

                if 'all_intermediate_features' not in locals():
                    all_intermediate_features    = np.zeros((len(dataloader)*dataloader.batch_size,y_intermediate_activations.view(y_intermediate_activations.shape[0],-1).shape[-1]))
                all_intermediate_features[iter*dataloader.batch_size:iter*dataloader.batch_size + dataloader.batch_size,:] = np.array(y_intermediate_activations.view(y_intermediate_activations.shape[0],-1).squeeze().cpu())
                all_labels[iter*dataloader.batch_size:iter*dataloader.batch_size + dataloader.batch_size] = np.array(labels.cpu()[:,np.newaxis])

            yhat = model(inputs)
            if args.extract_representation == True:
                all_predictions[iter*dataloader.batch_size:iter*dataloader.batch_size + dataloader.batch_size] = np.array(F.log_softmax(yhat).argmax(-1).cpu()[:,np.newaxis])
                all_output_activations[iter*dataloader.batch_size:iter*dataloader.batch_size + dataloader.batch_size] = np.array(F.softmax(yhat).cpu())
            loss = F.nll_loss(F.log_softmax(yhat), labels)

            sample_count += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)  # smaller batches count less
            running_acc += (yhat.argmax(-1) == labels).sum().item()  # num corrects
            current_cm = (running_tn, running_fp, running_fn, running_tp)
            running_tn, running_fp, running_fn, running_tp = metrics.update_confusionMatrix(labels,yhat,current_cm)

            _, yhat = yhat.topk(k, 1, True, True)
            running_acc_topk += (yhat == labels.view(-1, 1).expand_as(yhat)
                                 ).sum().item()  # num corrects

        loss = running_loss / sample_count
        acc = running_acc / sample_count
        top_k_acc = running_acc_topk / sample_count

        sn, sp, ppv, f1score = metrics.scores_from_confusionMatrix(running_tn, running_fp, running_fn, running_tp)
    if args.extract_representation == True:
        return loss, (acc, top_k_acc), sn, sp, ppv, f1score, all_intermediate_features, all_labels, all_predictions, all_output_activations
    else:
        return loss, (acc, top_k_acc), sn, sp, ppv, f1score
