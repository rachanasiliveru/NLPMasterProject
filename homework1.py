def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix for binary classification.
    Returns: TP, FP, FN, TN
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1
        else:
            tn += 1

    return tp, fp, fn, tn


def accuracy(y_true, y_pred):
    """
    Calculate accuracy: (correct predictions) / total
    Returns a float in [0, 1]
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    total = len(y_true)
    if total == 0:
        return 0.0

    correct = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct += 1

    return correct / total


def precision(y_true, y_pred):
    """
    Calculate precision: TP / (TP + FP)
    Must handle division-by-zero explicitly
    """
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(y_true, y_pred):
    """
    Calculate recall: TP / (TP + FN)
    Must handle division-by-zero explicitly
    """
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """
    Calculate F1 score:
    2 * precision * recall / (precision + recall)
    Must handle edge cases correctly
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    if prec + rec == 0:
        return 0.0

    return 2 * prec * rec / (prec + rec)


# ===== MULTI-CLASS SUPPORT =====

def unique_labels(y_true, y_pred):
    labels = []

    for label in y_true:
        if label not in labels:
            labels.append(label)

    for label in y_pred:
        if label not in labels:
            labels.append(label)

    try:
        return sorted(labels)
    except TypeError:
        return labels


def confusion_matrix_multiclass(y_true, y_pred):
    """
    Returns matrix[true_label][pred_label] = count
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    labels = unique_labels(y_true, y_pred)
    matrix = {}

    for true_label in labels:
        matrix[true_label] = {}
        for pred_label in labels:
            matrix[true_label][pred_label] = 0

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1

    return matrix


def precision_multiclass(y_true, y_pred, label):
    tp = 0
    fp = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label == label and true_label == label:
            tp += 1
        elif pred_label == label and true_label != label:
            fp += 1

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_multiclass(y_true, y_pred, label):
    tp = 0
    fn = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == label and pred_label == label:
            tp += 1
        elif true_label == label and pred_label != label:
            fn += 1

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_multiclass(y_true, y_pred, label):
    prec = precision_multiclass(y_true, y_pred, label)
    rec = recall_multiclass(y_true, y_pred, label)

    if prec + rec == 0:
        return 0.0

    return 2 * prec * rec / (prec + rec)


def macro_precision(y_true, y_pred):
    labels = unique_labels(y_true, y_pred)
    if len(labels) == 0:
        return 0.0

    total = 0.0
    for label in labels:
        total += precision_multiclass(y_true, y_pred, label)

    return total / len(labels)


def macro_recall(y_true, y_pred):
    labels = unique_labels(y_true, y_pred)
    if len(labels) == 0:
        return 0.0

    total = 0.0
    for label in labels:
        total += recall_multiclass(y_true, y_pred, label)

    return total / len(labels)


def macro_f1(y_true, y_pred):
    labels = unique_labels(y_true, y_pred)
    if len(labels) == 0:
        return 0.0

    total = 0.0
    for label in labels:
        total += f1_multiclass(y_true, y_pred, label)

    return total / len(labels)


if __name__ == "__main__":
    print("===== BINARY TESTS =====")

    # Test Case 1 — Perfect Classification
    y_true = [1, 1, 1, 0]
    y_pred = [1, 1, 1, 0]

    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
    print("Test Case 1 - Perfect Classification")
    print(f"TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print(f"accuracy = {accuracy(y_true, y_pred)}")
    print(f"precision = {precision(y_true, y_pred)}")
    print(f"recall = {recall(y_true, y_pred)}")
    print(f"f1 = {f1_score(y_true, y_pred)}")
    print()

    # Test Case 2 — All Wrong
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 1, 1]

    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
    print("Test Case 2 - All Wrong")
    print(f"TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print(f"accuracy = {accuracy(y_true, y_pred)}")
    print(f"precision = {precision(y_true, y_pred)}")
    print(f"recall = {recall(y_true, y_pred)}")
    print(f"f1 = {f1_score(y_true, y_pred)}")
    print()

    # Test Case 3 — No Predicted Positives
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]

    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
    print("Test Case 3 - No Predicted Positives")
    print(f"TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print(f"precision = {precision(y_true, y_pred)}")
    print(f"recall = {recall(y_true, y_pred)}")
    print(f"f1 = {f1_score(y_true, y_pred)}")
    print()

    # Test Case 4 — No Actual Positives
    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 0, 0]

    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
    print("Test Case 4 - No Actual Positives")
    print(f"TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print(f"precision = {precision(y_true, y_pred)}")
    print(f"recall = {recall(y_true, y_pred)}")
    print(f"f1 = {f1_score(y_true, y_pred)}")
    print()

    print("===== MULTI-CLASS TEST =====")

    y_true = ["cat", "dog", "cat", "bird", "dog", "bird", "cat"]
    y_pred = ["cat", "cat", "cat", "bird", "dog", "dog", "bird"]

    print("Accuracy =", accuracy(y_true, y_pred))
    print("Macro Precision =", macro_precision(y_true, y_pred))
    print("Macro Recall =", macro_recall(y_true, y_pred))
    print("Macro F1 =", macro_f1(y_true, y_pred))
    print("Confusion Matrix =", confusion_matrix_multiclass(y_true, y_pred))