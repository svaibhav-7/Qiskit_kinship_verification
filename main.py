import os
from src.models import FaceFeatureExtractor, QuantumKinshipModel
from src.data_loaders import load_kinfacew, load_tskinface
import numpy as np

def main():
    # Paths
    BASE = os.path.dirname(os.path.abspath(__file__))
    KFW1 = os.path.join(BASE, "KinFaceW-I", "KinFaceW-I")
    KFW2 = os.path.join(BASE, "KinFaceW-II")
    TSKIN = os.path.join(BASE, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")

    # Initialize model
    print("Initializing Quantum Kinship Model...")
    extractor = FaceFeatureExtractor()
    model = QuantumKinshipModel(n_qubits=4, n_layers=3, feature_extractor=extractor)

    # Load some data for testing
    print("Loading test data from KinFaceW-I...")
    try:
        X1_test, X2_test, Y_test = load_kinfacew(KFW1, extractor, n_qubits=4, max_pairs=10)
        print(f"Loaded {len(Y_test)} test pairs.")
        
        # Test one prediction
        score = model.predict(X1_test[0], X2_test[0])
        print(f"Prediction score for first pair: {score:.4f}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure datasets are in the correct paths.")

if __name__ == "__main__":
    main()
