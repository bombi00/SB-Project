import argparse
import os
import pandas as pd
from Bio.PDB import PDBList
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from NN import Predictor
import torch
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Predict features from a PDB file')
    parser.add_argument('--pdb', type=str, required=True, help='PDB code of the protein (e.g., 1ABC)')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file (.pkl for sklearn, .pth for NN)')
    parser.add_argument('--model_type', type=str, required=True, choices=['sklearn', 'nn'], help='Type of the model to use for prediction')
    return parser.parse_args()

def clean_features(df):
    categorical_cols = df.select_dtypes(include="object").columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[categorical_cols] = df[categorical_cols].fillna("missing")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def predict_features(pdb_code, model_path, model_type):
    pdb_code = pdb_code.upper()
    pdb_list = PDBList()
    
    print(f"Downloading PDB structure '{pdb_code}' in mmCIF format...")
    pdb_file = pdb_list.retrieve_pdb_file(pdb_code, pdir='./data/result/', file_format='mmCif')
    
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"Could not download structure for {pdb_code}.")

    print(f"Downloaded PDB file: {pdb_file}")
    
    print(f"{pdb_code} - extracting features...")
    exit_code = os.system(f"python3 ./data/script/calc_features.py {pdb_file} -out_dir ./data/result/")
    if exit_code != 0:
        raise RuntimeError("Feature extraction script failed")

    tsv_path = f'./data/result/{pdb_code.lower()}.tsv'
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Features file {tsv_path} not found.")

    original_features_df = pd.read_csv(tsv_path, sep='\t')
    processed_X = clean_features(original_features_df.copy())

    labels = {0: 'HBOND', 1: 'IONIC', 2: 'PICATION', 3: 'PIHBOND', 4: 'PIPISTACK', 5: 'SSBOND', 6: 'Unclassified', 7: 'VDW'}
    
    if model_type == 'sklearn':
        print("Using scikit-learn model for prediction...")
        model_bundle = joblib.load(model_path)
        clf = model_bundle["model"]
        expected_features = model_bundle["features"]

        feature_means = joblib.load('./data/model/feature_means.pkl')
        for col in expected_features:
            if col not in processed_X.columns:
                print(f"[INFO] Feature '{col}' mancante — riempita con media salvata")
                processed_X[col] = feature_means.get(col, 0)
        
        processed_X = processed_X[expected_features]

        output = clf.predict(processed_X)
        probs = clf.predict_proba(processed_X)
        probs_max = [f"{probs[i, o]:.4f}" for i, o in enumerate(output)]

    elif model_type == 'nn':
        print("Using Neural Network (PyTorch) model for prediction...")
        
        training_features = joblib.load('./data/model/training_features.pkl')
        
        feature_means = joblib.load('./data/model/feature_means.pkl')
        for col in training_features:
            if col not in processed_X.columns:
                print(f"[INFO] Feature '{col}' mancante — riempita con media salvata")
                processed_X[col] = feature_means.get(col, 0)
        
        print(f"Allineamento a {len(training_features)} feature di training.")
        processed_X = processed_X[training_features]

        input_dim = processed_X.shape[1] 
        output_dim = len(labels)
        
        model = Predictor(input_dim=input_dim, output_dim=output_dim)
        
        print(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        X_tensor = torch.tensor(processed_X.values, dtype=torch.float32)

        with torch.no_grad():
            logits = model(X_tensor)
            probabilities = F.softmax(logits, dim=1)
            probs_max_tensor, output_tensor = torch.max(probabilities, 1)
            output = output_tensor.numpy()
            probs_max = [f"{p:.4f}" for p in probs_max_tensor.numpy()]

    else:
        raise ValueError("model_type must be 'sklearn' or 'nn'")

    
    final_df = original_features_df.copy()
    
    # Verifica lunghezza corretta
    if len(output) != len(final_df):
        raise ValueError(f"Lunghezza predizioni ({len(output)}) diversa da numero righe ({len(final_df)})")

    final_df['PREDICTION'] = [labels[o] for o in output]
    final_df['PROBABILITY'] = probs_max
    
    output_filename = f'./data/result/{pdb_code}_predictions.tsv'
    final_df.to_csv(output_filename, sep='\t', index=False)
    print(f"\nResults saved to {output_filename}")


def main():
    args = parse_args()
    predict_features(args.pdb, args.model, args.model_type)

if __name__ == '__main__':
    main()