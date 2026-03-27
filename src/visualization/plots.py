import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as sk_metrics


def plot_training_history(hist, epochnum, batchnum, savepath, fullname):
    train_loss = hist['loss']
    val_loss = hist['val_loss']
    acc = hist['acc']
    val_acc = hist['val_acc']

    epochs = np.arange(1, len(train_loss) + 1, 1)

    plt.figure()
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('Loss, Model={}, Epochs={}, Batch={}'.format(fullname, epochnum, batchnum))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('{}/{}_Loss.png'.format(savepath, fullname), format='png')

    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('Accuracy, Model={}, Epochs={}, Batch={}'.format(fullname, epochnum, batchnum))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('{}/{}_Accuracy.png'.format(savepath, fullname), format='png')


def plot_roc_curve(y_true, predictions, model_name, savepath):
    fpr, tpr, thresh = sk_metrics.roc_curve(y_true, predictions)
    auc = sk_metrics.roc_auc_score(y_true, predictions)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label="{} = {:.4f}".format(model_name, auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(savepath, 'ROC.png'))
