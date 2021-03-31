import os

import torch
import numpy as np

from opensource.siamesetriplet.trainer import test_epoch

validation_file = "validation.txt"
result_folder = './result'

def validate_recognition_task(types, val_loader, model, validation_fn, cuda, testName, metrics=[]):
    validation_metrics = ["trueAccept", "trueReject", "falseAccept", "falseReject", "Accuracy", "svmAcc"]
    validation_memo = {name: [] for name in validation_metrics}



    val_loss, metrics = test_epoch(val_loader, model, validation_fn, cuda, metrics, testName+ "_"+types)
    message = "\n"
    for name, value in zip(metrics[0].name(), metrics[0].value()):
        validation_memo[name].append(value)
        message += '\t{}: {}'.format(name, value)
    with open(os.path.join(result_folder, types + validation_file), 'a') as file:
        file.write(message)
    print(message)
    # return training_loss, validation_memo
    return validation_memo

 
