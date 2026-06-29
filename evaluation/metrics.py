def safe_div(a, b):
    return a / b if b != 0 else 0.0


def detection_prf1(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1