# Deuterium

A fast, lightweight Python CLI that tells you whether an SMS message is **SPAM** (scammers offering you a "free" £900 prize) or **HAM** (actual humans asking about lunch).

No GPUs, no 40 GB neural networks, no cloud API keys. Just classic NLP (`TF-IDF` + `scikit-learn`) trained on the included `SMSSpamCollection` dataset in under a second.

---

## Quick Start

Pick your operating system, run the setup script once, and you're in business.

### Windows (PowerShell)
```powershell
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```
> **Note:** If PowerShell complains about execution policies, either run `setup.bat` from Command Prompt or run:
> `powershell -ExecutionPolicy Bypass -File .\setup.ps1`

### Windows (Command Prompt / CMD)
```bat
setup.bat
.venv\Scripts\activate.bat
```

### macOS / Linux
```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

What the setup script does for you:
1. Checks that you have Python 3.10+ (and gently judges you if you don't).
2. Creates an isolated `.venv/` virtual environment so your global Python stays clean.
3. Installs `scikit-learn`, `joblib`, `pytest`, and registers the `deuterium` CLI command.

---

## Usage

Setup intentionally **does not** train the model for you—you're in the driver's seat.

### 1. Train the Model
Train on the included `SMSSpamCollection` dataset (5,574 labeled messages):
```bash
deuterium train
```
This splits the data (80% train / 20% test, stratified), trains the classifier, prints accuracy/precision/recall/F1 + confusion matrix, and saves the model to `models/spam_model.joblib`.

Want to try Complement Naive Bayes instead of Logistic Regression?
```bash
deuterium train --algo nb
```

### 2. Predict a Message
Once trained, classify any message right from your terminal:
```bash
deuterium predict "WINNER!! You have been selected to receive a £900 prize reward! Call now!"
```
Output:
```text
Prediction: SPAM (98.4% confidence | spam=98.4%, ham=1.6%)
```

Or check a normal message:
```bash
deuterium predict "Hey, are we still meeting for lunch at noon?"
```
Output:
```text
Prediction: HAM (99.1% confidence | spam=0.9%, ham=99.1%)
```

Need machine-readable output for a script? Add `--json`:
```bash
deuterium predict "Free entry to win cash!" --json
```

### 3. Evaluate a Saved Model
Re-evaluate the saved model against `SMSSpamCollection` (or a custom tab-separated file):
```bash
deuterium evaluate
deuterium evaluate --data path/to/your_messages.tsv
```

> **Tip:** You can also invoke everything via `python -m deuterium ...` if you prefer not to activate `.venv` (e.g. `.\.venv\Scripts\python.exe -m deuterium predict "..."`).

---

## How the Model Works (Plain-English Edition)

1. **Text -> Numbers (`TfidfVectorizer`)**: Computers don't read English, so we convert each message into word and two-word phrase (`ngram_range=(1, 2)`) scores. Words that scream spam (*"free"*, *"claim"*, *"won prize"*) get high weights; words that show up everywhere (*"the"*, *"to"*) get dialed down.
2. **Classification (`LogisticRegression` or `ComplementNB`)**:
   - **`logreg` (default)**: Uses `class_weight="balanced"` because ~86.6% of the dataset is normal messages (`ham`). Without balancing, a lazy model could guess `ham` every time and still look 86.6% accurate on paper.
   - **`nb`**: `ComplementNB`, a Naive Bayes variant built specifically for imbalanced text datasets.

---

## Project Layout

```text
Deuterium/
├── SMSSpamCollection          # Labeled dataset (tab-separated: <ham|spam>\t<message>)
├── setup.ps1 / setup.bat      # Windows .venv setup scripts
├── setup.sh                   # macOS / Linux .venv setup script
├── pyproject.toml             # Package config & `deuterium` CLI entry point
├── requirements.txt           # Direct dependencies
├── src/deuterium/
│   ├── config.py              # Default file paths & constants
│   ├── data.py                # Dataset loader, validator, stratified splitter
│   ├── model.py               # ML pipeline, evaluation, save/load, prediction
│   ├── cli.py                 # `train`, `predict`, and `evaluate` commands
│   └── __main__.py            # Enables `python -m deuterium`
└── tests/                     # Pytest suite (data, model, and CLI tests)
```

---

## Running Tests

After running setup, run the test suite with:
```bash
pytest -v
```
