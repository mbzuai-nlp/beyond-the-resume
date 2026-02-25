# Beyond the Resume: A Rubric-Aware Automatic Interview System for Information Elicitation

<img width="2875" height="1793" alt="Screenshot 2026-02-24 154230" src="https://github.com/user-attachments/assets/08d764a8-2431-4245-8622-b773a4ddd9c9" />

This repository contains the code and dataset released for our paper on using Large Language Models as a low-cost proxy for subject matter experts (SMEs) during the early stage of hiring.

## Dataset

We release a dataset of resumes, belief calibration tests, and simulated interviews, all of which can be found under `data/`.

## Running the system

Firstly, ensure you have the Python manager, [uv](https://github.com/astral-sh/uv), installed in your system.

Setup the virtual environment using:

```
uv sync
```

Next, add a `.env` file to the project root with the following secret:

```
OPENAI_API_KEY=...
```

Finally, run the below command to spin up the system on `localhost:8000`:

```
uv run python -m chainlit run app/app_methods.py
```

## Reproducing our results (WIP)

This repository uses [DVC](https://dvc.org/) to define data pipelines. These pipelines can be run to reproduce our results...
