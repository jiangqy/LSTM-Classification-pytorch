import matplotlib.pyplot as plt
import pickle
from datetime import datetime

def PlotFigure(result, use_save=False):
    train_loss = result['train loss']
    test_loss = result['test loss']
    train_acc = result['train acc']
    test_acc = result['test acc']

    fig = plt.figure(1)

    font = {'family' : 'serif', 'color'  : 'black', 'weight' : 'bold', 'size'   : 16,}


    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(train_loss, 'r', label='Training Loss')
    ln2 = ax1.plot(test_loss, 'k', label='Testing Loss')
    ax2 = ax1.twinx()
    ln3 = ax2.plot(train_acc, 'r--', label='Training Accuracy')
    ln4 = ax2.plot(test_acc, 'k--', label='Testing Accuracy')

    lns = ln1+ ln2+ ln3+ ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=7)


    ax1.set_ylabel('Loss', fontdict=font)
    ax1.set_title("Text Classification", fontdict=font)
    ax1.set_xlabel('Epoch', fontdict=font)

    ax2.set_ylabel('Accuracy', fontdict=font)

    plt.show()
    if use_save:
        figname = 'figure/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pdf'
        fig.savefig(figname)
        print('Figure %s is saved.' % figname)
if __name__=='__main__':
    fp = open('log/LSTM_Classifier_0.pkl', 'rb')
    result = pickle.load(fp)
    PlotFigure(result, use_save=True)
