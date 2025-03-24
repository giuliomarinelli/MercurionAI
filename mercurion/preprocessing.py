import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import os
from mercurion.labels import tox21_labels

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    fp = generator.GetFingerprint(mol)
    bitstring = fp.ToBitString()
    return np.array([int(bit) for bit in bitstring], dtype=np.uint8)

def preprocess_tox21(input_csv='data/raw/tox21.csv',
                     output_X='data/processed/X.npy',
                     output_y='data/processed/y.npy'):
    df = pd.read_csv(input_csv)
    
    # Filtra righe con SMILES non validi
    df['fingerprint'] = df['smiles'].apply(smiles_to_fingerprint)
    df = df[df['fingerprint'].notnull()]

    X = np.stack(df['fingerprint'].values)
    y = df[tox21_labels].values.astype(np.float32)

    os.makedirs(os.path.dirname(output_X), exist_ok=True)
    np.save(output_X, X)
    np.save(output_y, y)

    print(f"Preprocessing completato. X shape: {X.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    preprocess_tox21()
