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
- A Jupyter notebook (`deep_learning.ipynb`) with the whole workflow: data
  checks including a near-duplicate check across the splits, tokenization, a
  TF-IDF reference model, the DistilBERT baseline with my own PyTorch
  classification head, the experiment with the class-weighted loss over five
  seeds, the evaluation with a paired t-test, results per agreement level,
  error examples and a short summary.
- A data preparation script (`prepare_dataset.py`) that turns the original
  archive into one CSV file with the agreement level, the template groups and
  the three splits.
- A written report (`Deep_Learning_Systems_Analysis_Report.pdf`) that
  explains the task, dataset, model, experiment, results, limitations and
  ethical risks for technical and non-technical readers. The project
  overview calls it `module_summary.pdf`, I use the file name from the
  submission instructions.
- A reproducibility file (`requirements.txt`), generated with `pip freeze`
  inside the project's own virtual environment.

## Dataset
Financial PhraseBank v1.0 (Malo et al., 2014), sentences from English
financial news about companies, labeled as negative, neutral or positive by
five to eight of 16 annotators with a background in finance, licensed CC
BY-NC-SA 3.0 for academic, non-commercial use:
https://huggingface.co/datasets/takala/financial_phrasebank

The archive `FinancialPhraseBank-v1.0.zip` (0.7 MB) holds the same sentences
at four levels of annotator agreement. It is not included, `prepare_dataset.py`
downloads it from a fixed revision and checks its SHA-256. The script removes
contradictory and repeated sentences, adds the agreement level and puts
sentences of the same template with other amounts or years into one group,
which gives 123 groups with more than one sentence. It splits the groups about
70/15/15 into train, validation and test, stratified by the label and
agreement level of the first sentence of each group, with seed 42. Templates
with larger changes can still land in two splits. I downloaded the archive on
2026-10-01. The result `phrasebank.csv` (0.7 MB) is included, one row per
sentence with the columns `sentence`, `label`, `agreement`, `group` and
`split`, with the original labels and under the same license, CC BY-NC-SA 3.0.

The notebook reads `phrasebank.csv` directly, so no download is needed to run
it. To rebuild the file from the original archive:
```
python prepare_dataset.py --download
```

This is the output of the script. The SHA-256 in the last line is the same
as the one of the file in this repository, which you can check with
`shasum -a 256 phrasebank.csv` (on Windows `certutil -hashfile phrasebank.csv SHA256`).
```
4,846 sentences in the archive, 4 with conflicting labels and 6 repeated removed, 4,836 kept
123 template groups with 338 sentences
train: 3,393 sentences, negative 12.6%, neutral 59.4%, positive 28.0%
validation: 721 sentences, negative 12.3%, neutral 59.6%, positive 28.0%
test: 722 sentences, negative 12.0%, neutral 58.9%, positive 29.1%
wrote phrasebank.csv, SHA-256 04af8c1589fbadd5dca774d53a49212bb35736d2d0698a1f82aa800cb2632a5d
```

## Main Results
Both configurations have 67.0 million parameters and were trained for 5 epochs
with five seeds (42 to 46), keeping the epoch with the best validation
macro-F1. On the 722 test sentences the baseline reaches a mean macro-F1 of
0.8176 (standard deviation 0.0043) and a negative recall of 0.8483. The
weighted loss raises the negative recall to 0.8644 and the macro-F1 to 0.8191
(0.0023), but lowers the negative precision from 0.7386 to 0.7294. The recall
is higher in only three of five seeds, and the mean changes are smaller than
the variation between the seeds (paired t-tests p = 0.605 for macro-F1,
p = 0.385 for the negative recall). So the weighted loss shifts the model a
little towards the negative class without making it clearly better or worse.
The TF-IDF reference reaches macro-F1 0.6483, or 0.6928 with balanced class
weights.

The clearest pattern is the agreement of the annotators. Where all annotators
agreed the models are right on 96% to 97% of the sentences, with only 50-65%
agreement on 57% to 59%. In the main run most errors sit between neutral and
positive.

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

On the first run the notebook downloads DistilBERT and its tokenizer from a
fixed revision on the Hugging Face hub (about 270 MB) and caches them. A full
run with ten models takes about one hour on the CPU of my Mac, CUDA is used
if it is available. The reported run used fixed seeds and deterministic
algorithms on the CPU, so results may differ slightly on other hardware.

## Dependencies
`requirements.txt` was created inside the project's own virtual
environment with:
```
pip freeze > requirements.txt
```

## Connection to Future AI Work
I stay in the domain of company data and compliance checks. Projects 1 to 3
built and analyzed company registers and a financial risk score, this project
adds a first step towards adverse media screening, finding the negative news
about a company. The labels measure investor sentiment, not compliance
relevance, so the model would need data labeled for that task. The planned
due diligence agent of the later capstone projects can use such a model to
sort news items before an analyst reads them. It should rank and flag, not
decide alone, and the results per agreement level show that unclear sentences
need a human reader most.
