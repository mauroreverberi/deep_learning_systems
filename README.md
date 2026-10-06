# Deep Learning Systems Project: Detecting Negative Company News


**GitHub repository:** https://github.com/mauroreverberi/deep_learning_systems
(all commits and the `develop` branch as well as the `main` branch are visible there)

## Project Description
I fine-tune a pretrained Transformer in PyTorch to read one sentence of
financial news about a company and decide whether it is negative, neutral
or positive for an investor. The output is one probability per class. As
the controlled comparison I train the same model a second time with one
change, a cross-entropy loss that weights each class by how rare it is, so
that the rare negative sentences count more. Both configurations are
trained with five seeds and evaluated on a held-out test split.

The question connects to my earlier capstone projects. In Project 1 I built a
cleaned dataset of Swiss legal entities from the GLEIF register, in Project 2
I analyzed new company registrations, and in Project 3 I turned a bankruptcy
score into a screening decision. A later capstone project will build a due
diligence agent that checks a company, and one standard step of such a check
is adverse media screening, reading the news about a company and flagging the
negative items.

## What Was Built
TBD

## Dataset
TBD

## Main Results
TBD

## How to Run the Project
1. Clone this repository.
2. Create and activate a virtual environment (Python 3.12 or newer, tested with 3.14):
   ```
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Start Jupyter:
   ```
   jupyter lab
   ```
5. Open `deep_learning.ipynb` and run all cells
   (Kernel > Restart Kernel and Run All Cells).


## Dependencies
`requirements.txt` was created inside the project's own virtual
environment with:
```
pip freeze > requirements.txt
```

## Connection to Future AI Work
TBD
