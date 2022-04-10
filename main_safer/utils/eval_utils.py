from sklearn.metrics import f1_score


def acc_f1(output, labels, average='binary'):
    predictions = output.max(1)[1].type_as(labels)
    if predictions.is_cuda:
        predictions = predictions.cpu()
        labels = labels.cpu()
    return f1_score(predictions, labels, average=average)
