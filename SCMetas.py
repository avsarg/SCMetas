# -*- coding: utf-8 -*-
"""
@author: Gülben Avşar
Prediction file
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from pydeseq2.dds import DeseqDataSet


# ==========================================
# 1. ANN ARCHITECTURE
# ==========================================

class ANN(nn.Module):


    def forward(self, x):

        return self.network(x)

# ==========================================
# 2. LOAD MODEL ONCE AT IMPORT
# ==========================================

_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_MODEL_PATH = Path(__file__).resolve().parent / "SCMetas.pth"

_PACKAGE = torch.load(
    _MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

_ARCHITECTURE = _PACKAGE["architecture"]

_MODELS = []

for weights in _PACKAGE["model_weights"]:

    model = ANN(**_ARCHITECTURE)

    model.load_state_dict(weights)

    model.to(_DEVICE)
    model.eval()

    _MODELS.append(model)

_SCALERS = _PACKAGE["scalers"]

print("SCMetas loaded successfully!")


# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================

def SCMetas(rawdata, metadata):

    """
    SCMetas: Five-fold ANN classifier.

    Parameters
    ----------
    rawdata : dataframe
        rows = samples, columns = genes
    
    metadata : dataframe
        metadata indicates the sample classes
        rows = samples, columns = experimental conditions

    Returns
    -------
    pandas DataFrame
        Normalized and transformed data, predicted classes and
        class probabilities.
    """

    # preprocess input data
    data = preprocess(rawdata, metadata)

    X = data.copy()

    all_probabilities = []

    # Obtain predictions from five models
    for model, scaler in zip(_MODELS, _SCALERS):

        X_scaled = scaler.transform(X)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        ).to(_DEVICE)

        with torch.no_grad():

            logits = model(X_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    # Average probabilities
    ensemble_probabilities = np.mean(
        all_probabilities,
        axis=0
    )

    # Determine predicted classes
    predicted_classes = np.argmax(
        ensemble_probabilities,
        axis=1
    )

    # Prepare results
    results = data.copy()

    results["Predicted_Class"] = predicted_classes

    for i in range(3):

        results[f"Probability_Class_{i}"] = (
            ensemble_probabilities[:, i]
        )

    return results


def preprocess(rawdata, metadata):
    # Read input data
    data = pd.read_csv(rawdata)

    X = data.copy()
    
    genes = pd.read_csv('panelgenes.txt')
    s1 = pd.Series(genes.Gene)
    s2 = pd.Series(X.columns)
    common = s1[s1.isin(s2)]
    print('The number of common genes: ', len(common))

    
    counts_df = data.copy() 

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~ 1",  # Use an empty intercept design for unsupervised HVG selection
        )

    vst_matrix = pd.DataFrame(
        dds.layers["vst_counts"], 
        index=counts_df.index, 
        columns=counts_df.columns)


    data = vst_matrix.copy()
    
    return(data)





