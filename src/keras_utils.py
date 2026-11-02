"""Small Keras helpers shared by the experiments."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def evaluate_classifier(model, X, y, labels):
    if getattr(X, "dtype", None) == np.uint8:          # images: cast batch-wise to save RAM
        import tensorflow as tf
        X = tf.data.Dataset.from_tensor_slices(X).batch(256).map(lambda a: tf.cast(a, tf.float32))
        prob = model.predict(X, verbose=0)
    else:
        prob = model.predict(X, verbose=0, batch_size=1024)
    pred = prob.argmax(1)
    return {
        "test_accuracy": round(float(accuracy_score(y, pred)), 4),
        "test_macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(labels))).tolist(),
        "report": classification_report(y, pred, target_names=labels, output_dict=True, zero_division=0),
    }


def dense_layers_to_json(model):
    """Export Dense / BatchNormalization layers in order, for the in-browser demo."""
    layers = []
    for layer in model.layers:
        cls = layer.__class__.__name__
        w = layer.get_weights()
        if cls == "Dense":
            bias = w[1] if len(w) > 1 else np.zeros(w[0].shape[1], np.float32)   # use_bias=False before BN
            layers.append({"type": "dense", "activation": layer.get_config()["activation"],
                           "W": np.round(w[0], 6).tolist(), "b": np.round(bias, 6).tolist()})
        elif cls == "BatchNormalization":
            gamma, beta, mean, var = w
            layers.append({"type": "bn", "gamma": np.round(gamma, 6).tolist(), "beta": np.round(beta, 6).tolist(),
                           "mean": np.round(mean, 6).tolist(), "var": np.round(var, 6).tolist(),
                           "eps": layer.get_config()["epsilon"]})
        elif cls == "Activation":
            layers.append({"type": "activation", "activation": layer.get_config()["activation"]})
    return layers
