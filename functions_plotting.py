import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import NullFormatter
from sklearn.calibration import calibration_curve

def confusion_matrix_with_marginals(true_scores, predicted_scores, title='Confusion matrix'):
    """
    Plots confusion matrix for predicted scores (x axis) and true scores (y axis) with marginal histograms for true scores and predicted scores on corresponding sides. 
    Pass title as title = ''Confusion matrix for the ordinal regression - test data' to be more descriptive based on whether the plot is for test or train data
    """
    plt.figure(figsize=(20, 8))

    # define axes
    ax_Pxy = plt.axes((0.2, 0.34, 0.27, 0.52))
    ax_Px = plt.axes((0.2, 0.14, 0.27, 0.2))
    ax_Py = plt.axes((0.1, 0.34, 0.1, 0.52))


    ax_Pxy.xaxis.set_major_formatter(NullFormatter())
    ax_Pxy.yaxis.set_major_formatter(NullFormatter())

    ax_Px.yaxis.set_major_formatter(NullFormatter())
    # ax_Py.xaxis.set_major_formatter(NullFormatter())

    # draw the joint probability
    plt.axes(ax_Pxy)

    conf_matrix = confusion_matrix(true_scores.int(), predicted_scores.int())
    max_score = true_scores.max().int()
    df_cm = pd.DataFrame(conf_matrix, range(max_score + 1), range(max_score + 1))
    sns.set(font_scale=1.8) # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g', cbar=False) # font size fmt = 'g' to do with unscaled option

    plt.xlabel('predicted score')
    plt.ylabel('true score')

    # draw predicted scores marginal distribution
    sns.histplot(predicted_scores, ax=ax_Px, color='red')
    ax_Px.set_xlabel('predicted score')

    # draw true scores marginal distribution
    sns.histplot(y=true_scores, ax=ax_Py, color='green')
    ax_Py.invert_yaxis()
    ax_Py.set_ylabel('true score')


    # label axes
    ax_Pxy.set_xlabel('predicted score', fontsize = 14)
    ax_Pxy.set_ylabel('true score', fontsize = 14)


    ax_Pxy.set_title(title)
    ax_Pxy.set_xlabel('predicted score')
    ax_Pxy.set_ylabel('true score')


    plt.show()

def plot_confusion_matrix(true_scores, predicted_scores, title = 'Confusion matrix'):
    conf_matrix = confusion_matrix(true_scores.int(), predicted_scores.int())
    df_cm = pd.DataFrame(conf_matrix, range(6), range(6))
    plt.figure(figsize=(13,8))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size fmt = 'g' to do with unscaled option
    plt.title(title)
    plt.xlabel('predicted score', fontsize = 14)
    plt.ylabel('true score', fontsize = 14)
    plt.show()


def plot_calibration_curve(test_data, prob_matrix, figsize = (13, 8), title = None):
    test_scores = test_data[:, 2]

    plt.figure(figsize=figsize)

    for k in range(6):
        test_scores_binarised = (test_scores == k).int()
        y_true = test_scores_binarised 
        y_pred = prob_matrix[:, k]
        y_pred = y_pred.detach().numpy()
        fop, mpv = calibration_curve(y_true, y_pred, n_bins=20, normalize=True)
        l='true score ' + str(k)
        plt.plot(mpv, fop, marker='.', label=l)

    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', label = 'calibration line')
    # plot model reliability
    if not title:
        title = 'Calibration plot' 
    
    plt.legend(fontsize = 14)
    plt.title(title, fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xlabel('predicted probability', fontsize = 16)
    plt.ylabel('true probability in each bin', fontsize = 16)
    plt.show()