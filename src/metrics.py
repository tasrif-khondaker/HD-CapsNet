import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix, multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay, average_precision_score
from scipy.stats import hmean
from treelib import Tree
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import os


def get_top_k_accuracy_score(y_true: list, y_pred: list, k=1):
    if len(list(y_pred[0])) == 2:
        if k == 1:
            return accuracy_score(y_true, np.argmax(y_pred, axis=1))
        else:
            return 1
    else:
        return top_k_accuracy_score(y_true, y_pred, k=k)


def get_top_k_taxonomical_accuracy(y_true: list, y_pred: list, k=1):
    """
    This method computes the top k accuracy for each level in the taxonomy.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    if len(y_true) != len(y_pred):
        raise Exception('Size of the inputs should be the same.')
    accuracy = [get_top_k_accuracy_score(y_, y_pred_, k) for y_, y_pred_ in zip(y_true, y_pred)]
    return accuracy


def get_h_accuracy(y_true: list, y_pred: list, k=1):
    """
    This method computes the harmonic mean of accuracies of all level in the taxonomy.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    return hmean(get_top_k_taxonomical_accuracy(y_true, y_pred, k))


def get_m_accuracy(y_true: list, y_pred: list, k=1):
    """
    This method computes the arithmetic mean of accuracies of all level in the taxonomy.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    return np.mean(get_top_k_taxonomical_accuracy(y_true, y_pred, k))


def get_exact_match(y_true: list, y_pred: list):
    """
    This method compute the exact match score. Exact match is defined as the #of examples for
    which the predictions for all level in the taxonomy is correct by the total #of examples.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: the exact match value
    :rtype: float
    """
    y_pred = [np.argmax(x, axis=1) for x in y_pred]
    if len(y_true) != len(y_pred):
        raise Exception('Shape of the inputs should be the same')
    exact_match = []
    for j in range(len(y_true[0])):
        v = 1
        for i in range(len(y_true)):
            if y_true[i][j] != y_pred[i][j]:
                v = 0
                break
        exact_match.append(v)
    return np.mean(exact_match)


def get_consistency(y_pred: list, tree: Tree):
    """
    This methods estimates the consistency.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: np.array
    :param tree: A tree of the taxonomy.
    :type tree: Tree
    :return: value of consistency.
    :rtype: float
    """
    y_pred = [np.argmax(x, axis=1) for x in y_pred]
    consistency = []
    for j in range(len(y_pred[0])):
        v = 1
        for i in range(len(y_pred) - 1):
            parent = 'L' + str(i) + '_' + str(y_pred[i][j])
            child = 'L' + str(i + 1) + '_' + str(y_pred[i + 1][j])
            if tree.parent(child).identifier != parent:
                v = 0
                break
        consistency.append(v)
    return np.mean(consistency)


def get_mAP_Score(y_true: list, y_pred: list):
    if len(y_true) != len(y_pred):
        raise Exception('Size of the inputs should be the same.')
    # print(y_pred.shape)
    mAP_score = [average_precision_score(y_, y_pred_) for y_, y_pred_ in zip(y_true, y_pred)]
    return mAP_score

def get_h_mAP_score(y_true: list, y_pred: list):
    """
    This method computes the mean Average precision for all the hierarchical levels and take the harmonic mean.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    return hmean(get_mAP_Score(y_true, y_pred))

def get_m_mAP_score(y_true: list, y_pred: list):
    """
    This method computes the mean Average precision for all the hierarchical levels and take the arithmetic mean.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    return np.mean(get_mAP_Score(y_true, y_pred))


def get_hierarchical_metrics(y_true: list, y_pred: list, tree: Tree):
    """
    This method compute the hierarchical precision/recall/F1-Score. For more details, see:

    Kiritchenko S., Matwin S., Nock R., Famili A.F. (2006) Learning and Evaluation
    in the Presence of Class Hierarchies: Application to Text Categorization. In: Lamontagne L.,
    Marchand M. (eds) Advances in Artificial Intelligence. Canadian AI 2006. Lecture Notes in
    Computer Science, vol 4013. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11766247_34

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :param tree: A tree of the taxonomy.
    :type tree: Tree
    :return: the hierarchical precision/recall/F1-Score values
    :rtype: float
    """
    y_pred = [np.argmax(x, axis=1) for x in y_pred]

    if len(y_true) != len(y_pred):
        raise Exception('Shape of the inputs should be the same')
    hP_list = []
    hR_list = []
    hF1_list = []
    for j in range(len(y_true[0])):
        y_true_aug = set()
        y_pred_aug = set()
        for i in range(len(y_true)):
            true_c = 'L' + str(i) + '_' + str(y_true[i][j])
            y_true_aug.add(true_c)
            while tree.parent(true_c) != None:
                true_c = tree.parent(true_c).identifier
                y_true_aug.add(true_c)

            pred_c = 'L' + str(i) + '_' + str(y_pred[i][j])
            y_pred_aug.add(pred_c)
            while tree.parent(pred_c) != None:
                pred_c = tree.parent(pred_c).identifier
                y_pred_aug.add(pred_c)

        y_true_aug.remove('root')
        y_pred_aug.remove('root')

        hP = len(y_true_aug.intersection(y_pred_aug)) / len(y_pred_aug)
        hR = len(y_true_aug.intersection(y_pred_aug)) / len(y_true_aug)
        if 2 * hP + hR != 0:
            hF1 = 2 * hP * hR / (hP + hR)
        else:
            hF1 = 0

        hP_list.append(hP)
        hR_list.append(hR)
        hF1_list.append(hF1)
    return np.mean(hP_list), np.mean(hR_list), np.mean(hF1_list)


def performance_report(y_true: list, y_pred: list, tree: Tree, title=None):
    """
        Build a text report showing the main classification metrics.

        :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
        :type y_pred: list
        :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
        :type y_true: list
        :param tree: A tree of the taxonomy.
        :type tree: Tree
        :param title: A title for the report.
        :type title: str
        :return: the hierarchical precision/recall/F1-Score values
        :rtype: float
        """
    y_true_argmax=[[] for x in range(len(y_true))]
    for x in range(len(y_true)):
        y_true_argmax[x]=np.argmax(y_true[x], axis=1).tolist()
    accuracy = get_top_k_taxonomical_accuracy(y_true_argmax, y_pred)
    exact_match = get_exact_match(y_true_argmax, y_pred)
    consistency = get_consistency(y_pred, tree)
    hP, hR, hF1 = get_hierarchical_metrics(y_true_argmax, y_pred, tree)
    HarmonicM_Accuracy_k1 = get_h_accuracy(y_true_argmax, y_pred, k=1)
    HarmonicM_Accuracy_k2 = get_h_accuracy(y_true_argmax, y_pred, k=2)
    HarmonicM_Accuracy_k5 = get_h_accuracy(y_true_argmax, y_pred, k=5)
    ArithmeticM_Accuracy_k1 = get_m_accuracy(y_true_argmax, y_pred, k=1)
    ArithmeticM_Accuracy_k2 = get_m_accuracy(y_true_argmax, y_pred, k=2)
    ArithmeticM_Accuracy_k5 = get_m_accuracy(y_true_argmax, y_pred, k=5)
    Harmonic_mAP_Score = get_h_mAP_score(y_true, y_pred)
    Arithmetic_mAP_Score = get_m_mAP_score(y_true, y_pred)

    out={}

    row = []
    for i in range(len(accuracy)):
        row.append('Accuracy L_' + str(i))
        row.append("{:.4f}".format(accuracy[i]))
        out['Accuracy L_' + str(i)] = accuracy[i]
    out = {**out, **{
                    'HarmonicM_Accuracy_k1': HarmonicM_Accuracy_k1,
                    'HarmonicM_Accuracy_k2': HarmonicM_Accuracy_k2,
                    'HarmonicM_Accuracy_k5': HarmonicM_Accuracy_k5,
                    'ArithmeticM_Accuracy_k1': ArithmeticM_Accuracy_k1,
                    'ArithmeticM_Accuracy_k2': ArithmeticM_Accuracy_k2,
                    'ArithmeticM_Accuracy_k5': ArithmeticM_Accuracy_k5,
                    'Harmonic_mAP_Score': Harmonic_mAP_Score,
                    'Arithmetic_mAP_Score': Arithmetic_mAP_Score,
                    'hP': hP, 
                    'hR': hR, 
                    'hF1': hF1,
                    'consistency': consistency,
                    'exact_match': exact_match
                    }
           }
    
    return out



def lvl_wise_metric(y_true: list, y_pred: list,savedir:str=None,show_graph:bool=True):
    if len(y_true) < 1:
        raise ValueError("Invalid length of y_true. At least one level is required.")

    level_classes_numbers = [len(np.unique(np.argmax(y, axis=1))) for y in y_true]
    level_labels = [list(range(0, num_classes)) for num_classes in level_classes_numbers]
    level_target_names = [[str(x) for x in range(0, num_classes)] for num_classes in level_classes_numbers]

    for level, (y_true_level, y_pred_level) in enumerate(zip(y_true, y_pred)):
        print('\033[91m', '\033[1m', "\u2022", f'Confusion_Matrix Level = {level}', '\033[0m')
        # print(confusion_matrix(np.argmax(y_true_level, axis=1), np.argmax(y_pred_level, axis=1)))
        confusion_matrixDisplay(np.argmax(y_true_level, axis=1), np.argmax(y_pred_level, axis=1), level_target_names[level],savedir,show_graph)

        print('\n\033[91m', '\033[1m', "\u2022", f'Classification Report for Level = {level}', '\033[0m\n')
        
        print(
            classification_report(
                np.argmax(y_true_level, axis=1),
                np.argmax(y_pred_level, axis=1),
                target_names=level_target_names[level],
                digits=5
            )
        )

def confusion_matrixDisplay(y_true, y_pred, target_names,savedir,show_graph):
    size_of_fig = len(target_names)/2
    # size_of_fig = 100
    if size_of_fig < 7:
        size_of_fig = 7
    labels = target_names
    cm = confusion_matrix(y_true, y_pred)
    cmp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(size_of_fig,size_of_fig))
    cmp.plot(ax=ax)
    if savedir is not None:
        dpi_val = 1080/size_of_fig
        plt.savefig(os.path.join(savedir,f'LVL_Len_{len(target_names)}.png'), dpi=dpi_val)
    if show_graph is True:
        plt.show()

def hmeasurements(y_true: list,
                  y_pred: list,
                  tree):
    y_true_argmax=[[] for x in range(len(y_true))]
    for x in range(len(y_true)):
        y_true_argmax[x]=np.argmax(y_true[x], axis=1).tolist()
    h_measurements = get_hierarchical_metrics(y_true_argmax,y_pred,tree)
    consistency = get_consistency(y_pred, tree)
    exact_match = get_exact_match(y_true_argmax, y_pred)
    get_performance_report = performance_report(y_true, y_pred, tree)
    return h_measurements,consistency,exact_match, get_performance_report
    
    
if __name__ == '__main__':
    y = [[1, 0, 1, 0, 0], [1, 2, 3, 4, 0], [3, 4, 5, 8, 0]]

