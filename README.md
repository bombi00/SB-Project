# Protein Interaction Predictor

This tool allows you to **predict molecular interactions** from a PDB code using either a **scikit-learn** or **Neural Network (PyTorch)** model.

## Install Dependencies

1. Open a terminal.
2. Navigate to the main folder of the project.
3. Install required packages with:

```bash
pip install -r requirements.txt
```

## Predict Interactions

To make predictions, run the following command from the main project folder:

```bash
python3 predictor.py --pdb YOUR_PDB_CODE --model_type [sklearn|nn] --model ./data/model/[model.pkl|weights.pth]
```

### Arguments:

- `--pdb`: The PDB code of the protein to analyze (e.g. `1ABC`)
- `--model_type`: Choose the type of model:
  - `sklearn` → for scikit-learn models (e.g. HistGradientBoosting)
  - `nn` → for neural network models (PyTorch)
- `--model`: Path to the trained model file:
  - `model.pkl` for sklearn
  - `weights.pth` for PyTorch

## Example Usage

```bash
python3 predictor.py --pdb 1ABC --model_type sklearn --model ./data/model/hg_model.pkl
```

or

```bash
python3 predictor.py --pdb 2XYZ --model_type nn --model ./data/model/model_weights.pth
```

## Output

- Results will be saved as a `.tsv` file in `./data/result/` named:

```
data/result/YOUR_PDB_predictions.tsv
```

This file includes:
- Residue information
- Predicted interaction type
- Prediction confidence (probability)

## Re-train the model

To re-train the model is necessary to execute all the cell in the `Data_preprocessing.ipnyb`