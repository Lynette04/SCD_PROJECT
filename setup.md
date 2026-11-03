# Setting up the PyTorch version

These steps are for Windows and use PowerShell. Run every command from inside the project folder (the one that contains `app.py`).

The project now uses PyTorch instead of TensorFlow. The models, training settings, data splits and scripts are the same as before. The changes are listed in [What changed](#what-changed) at the end.

## 1. Check your Python version

```powershell
py --list
```

You need Python **3.12 or 3.13**. If neither is listed, install Python 3.13 from [python.org](https://www.python.org/downloads/windows/).

## 2. Save your work and pull the update

Open PowerShell in your project folder, then check for unsaved work:

```powershell
git status
```

If it lists changed files, commit them first. Your own `.keras` files won't be touched, even if they aren't tracked by Git.

```powershell
git switch main
git pull
```

The TensorFlow version is still in the project's history. To look at it, run `git switch --detach 619bec8`. To come back, run `git switch main`.

## 3. Create a virtual environment

This keeps the project's packages separate from everything else on your laptop.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Use `py -3.12` instead if that's the version you have.

If PowerShell says running scripts is disabled, run this once and then try the activate line again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

When it's active, you'll see `(.venv)` at the start of the line. Activate it again each time you open a new PowerShell window.

## 4. Install the packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs about 1 GB of packages, so it can take a while.

## 5. Check that it works

```powershell
python step2_data_pipeline.py
python measure_mobilenet_share.py
```

The first command should print:

```text
Class names: ['normal', 'sickle']
Training image counts: {'normal': 102, 'sickle': 295}
```

Its last lines also show the shape of a batch, `[16, 3, 224, 224]`. The colour channel comes first now because that's how PyTorch stores images.

The second command should end with:

```text
Backbone trainable weights:1510720 of 2257984 = 66.9 percent
```

## 6. Check the converted models

All 5 trained models (3 MobileNetV2 and 2 EfficientNet-Lite0) are already converted and included as `.pt` files. To check the EfficientNet ones:

```powershell
python eval_effnet_lite0.py
```

If anyone retrains a model with the old TensorFlow code and commits a new `.keras` file, run this to update the `.pt` files from it:

```powershell
python convert_keras_models.py
```

Any `.keras` file it can't find is skipped with a message.

## 7. Run the scripts

You only need these if you want to retrain. Each one saves its model as a `.pt` file and **replaces** the existing file with the same name, including your converted models.

| Order | Command | What it does | Saves |
| --- | --- | --- | --- |
| 1 | `python step3_train_model.py` | Trains MobileNetV2 | `sickle_cell_model.pt`, `loss_curves.png`, `training_loss.png` |
| 2 | `python step4_finetune_model.py` | Fine-tunes MobileNetV2 | `Sickle_cell_fine_tuned-mode.pt`, `training_loss_finetune.png` |
| 3 | `python step5_train_edge_model.py` | Trains the small edge model | `sickle_cell_edge_model.pt`, `training_loss_edge.png` |
| 4 | `python train_efficientnet_lite0.py` | Trains EfficientNet-Lite0 | `sickle_cell_effnet_lite0.pt` |
| 5 | `python finetune_efficientnet_lite0.py` | Fine-tunes EfficientNet-Lite0 | `sickle_cell_effnet_lite0_ft30.pt` |
| 6 | `python eval_effnet_lite0.py` | Prints the confusion matrix | Nothing |
| 7 | `python inspect_effnet.py` | Lists the EfficientNet layers | Nothing |

Without a GPU, expect each training script to take from about 10 minutes to about an hour. EfficientNet fine-tuning is the slowest. The two fine-tuning scripts continue from the model trained just before them, so keep the order in the table. The ImageNet weights download automatically the first time, so you need internet access.

You don't need `step1_prepare_data.py`, because the `data` folder is already in the project. `check_split.py` and `app.py` work as before:

```powershell
python check_split.py
python app.py
```

The web page is at <http://127.0.0.1:5000>. As before, `app.py` only receives the uploaded file and doesn't use a model yet.

The loss plots show the training and validation loss for each epoch, with a dashed line on the best epoch.

`augmentation_check.py`, `eval_all_models.py` and `check_quantisation.py` still use TensorFlow and the `.keras` files, so they don't run in this environment.

## What changed

- **Models:** the files are now `.pt` instead of `.keras`. The original `.keras` files are still in the project, and `convert_keras_models.py` uses them.
- **New files:**
  - `pytorch_mobilenet.py` and `pytorch_efficientnet.py` hold the models.
  - `pytorch_training.py` holds the code that replaces `compile`, `fit`, `evaluate` and `save`.
- **Layer order:** every layer keeps its Keras name and position, so "the last 30 layers" still means the same layers.
- **Augmentation:** it follows the same formulas as the Keras layers, because the built-in PyTorch versions work differently.
- **Fixed file name:** `finetune_efficientnet_lite0.py` used to save to `sickle_celll_...` with 3 l's, so `eval_effnet_lite0.py` couldn't find the model. It now saves to `sickle_cell_effnet_lite0_ft30.pt`.
- **Loss plots:** steps 3, 4 and 5 save a plot of the loss curves through `plot_loss` in `pytorch_training.py`.

## Results to expect

The converted models give the same accuracy, precision and recall as the Keras originals. The loss differs in the third decimal place, because PyTorch's image loading decodes JPEGs slightly more accurately than TensorFlow's.

Retrained models will give slightly different numbers from your Keras results. Shuffling and augmentation are random, and PyTorch can't reproduce TensorFlow's random numbers. The test set has 87 images, so one image changes accuracy by about 1%.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named torch` | The virtual environment isn't active. Run `.\.venv\Scripts\Activate.ps1`. |
| `No matching distribution found for torch` | Your Python version isn't supported. Use 3.12 or 3.13. |
| `FileNotFoundError` for a `.pt` file | Run the script that creates it (step 7), or convert it (step 6). |
| `FileNotFoundError` for `data/train` | You're in the wrong folder. Go to the one that contains `app.py`. |
| A warning about `HF_TOKEN` | You can ignore this. It only affects download speed. |
