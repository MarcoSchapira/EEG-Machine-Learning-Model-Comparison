import os
import sys

import numpy as np
import torch


def _import_model_module_for_unpickle(model_name):
    """Import only the wrapper module pickle needs; TCNet pulls torcheeg/torch_scatter."""
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    nested = os.path.normpath(os.path.join(_here, "..", "EEG Model Training"))
    if not os.path.isfile(os.path.join(_here, "MSCFormerModel.py")) and os.path.isfile(
        os.path.join(nested, "MSCFormerModel.py")
    ):
        if nested not in sys.path:
            sys.path.insert(0, nested)
    if model_name == "MSCFormer":
        import MSCFormerModel  # noqa: F401
    elif model_name == "EEGEncoder":
        import EEGEncoderModel  # noqa: F401
    elif model_name == "TCNet":
        import TCNet_Model  # noqa: F401
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def _torch_load_trusted(path):
    """Local checkpoints are full pickles; PyTorch 2.6+ defaults weights_only=True."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def inference_device():
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_sample(eeg_sample, model_name, device):
    sample_mean = float(np.mean(eeg_sample))
    sample_std = float(np.std(eeg_sample))
    if sample_std < 1e-8:
        sample_std = 1e-8
    eeg_sample = (eeg_sample - sample_mean) / sample_std
    tensor_sample = torch.from_numpy(eeg_sample).float().to(device)
    if model_name in ("MSCFormer", "EEGEncoder"):
        tensor_sample = tensor_sample.unsqueeze(0).unsqueeze(0)
    elif model_name == "TCNet":
        tensor_sample = tensor_sample.unsqueeze(0)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    return tensor_sample


def predict_class(model, eeg_sample_np, model_name, device):
    x = prepare_sample(eeg_sample_np, model_name, device)
    with torch.no_grad():
        _, logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
    return pred


def load_dataset_tensors(data_path):
    bundle = _torch_load_trusted(data_path)
    if isinstance(bundle, dict) and "x_test" in bundle and "y_test" in bundle:
        x, y = bundle["x_test"], bundle["y_test"]
    elif isinstance(bundle, dict) and "data" in bundle and "label" in bundle:
        x, y = bundle["data"], bundle["label"]
    else:
        raise KeyError(
            "Expected .pt dict with ('x_test','y_test') or ('data','label'); "
            f"got keys: {list(bundle.keys()) if isinstance(bundle, dict) else type(bundle)}"
        )
    return x, y


def evaluate_all_trials(model, eeg_data, eeg_labels, model_name, device, class_names):
    labels_np = eeg_labels.cpu().numpy() if torch.is_tensor(eeg_labels) else np.asarray(eeg_labels)
    n_classes = len(class_names)
    print(f"\nPer-class results (true label vs model prediction), device={device}:\n")
    total_trials = 0
    total_correct = 0

    for class_idx in range(n_classes):
        indices = np.where(labels_np == class_idx)[0]
        n_trials = int(len(indices))
        n_correct = 0
        for i in indices:
            sample = eeg_data[int(i)]
            if torch.is_tensor(sample):
                sample = sample.detach().cpu().numpy()
            pred = predict_class(model, sample, model_name, device)
            if pred == class_idx:
                n_correct += 1
        total_trials += n_trials
        total_correct += n_correct
        name = class_names[class_idx]
        print(f"  {name}: trials={n_trials}, correct={n_correct}")

    acc = 100.0 * total_correct / total_trials if total_trials else 0.0
    print(f"\nOverall: trials={total_trials}, correct={total_correct}, accuracy={acc:.2f}%\n")


def main():
    class_names = [
        "Arm Forward",
        "Arm Backward",
        "Arm Left",
        "Arm Right",
        "Arm Up",
        "Arm Down",
        "Hand Grasping",
        "Wrist Rotation",
        "Rest",
    ]

    MODEL_ARCHITECTURE = "MSCFormer"
    _here = os.path.dirname(os.path.abspath(__file__))
    _ben_data = os.path.normpath(os.path.join(_here, "..", "Test_WithBenData"))
    if MODEL_ARCHITECTURE == "MSCFormer":
        model_path = os.path.join(_ben_data, "MSCFormer_model_sub1_27node_Production.pth")
    elif MODEL_ARCHITECTURE == "EEGEncoder":
        model_path = os.path.join(_ben_data, "EEGEncoder_model_sub1_27node_Production.pth")
    elif MODEL_ARCHITECTURE == "TCNet":
        model_path = os.path.join(_ben_data, "TCNet_model_sub1_27node_Production.pth")
    else:
        raise ValueError(f"Unknown model architecture: {MODEL_ARCHITECTURE}")
    data_path = os.path.join(_ben_data, "EEG_Ben.pt")
    channels_to_keep = list(range(27))

    device = inference_device()
    if device.type == "cuda":
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print(f"Loading {MODEL_ARCHITECTURE} from {model_path}...")
    print(f"Using torch device: {device}")
    _import_model_module_for_unpickle(MODEL_ARCHITECTURE)
    model = _torch_load_trusted(model_path).to(device)
    model.eval()

    print(f"Loading data from {data_path}...")
    eeg_data, eeg_labels = load_dataset_tensors(data_path)
    if channels_to_keep is not None:
        eeg_data = eeg_data[:, channels_to_keep, :]
        print(f"Filtered channels. Data shape: {list(eeg_data.shape)}")

    evaluate_all_trials(model, eeg_data, eeg_labels, MODEL_ARCHITECTURE, device, class_names)


if __name__ == "__main__":
    main()
