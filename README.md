# Beyond the Resume: A Rubric-Aware Automatic Interview System for Information Elicitation

<img width="2875" height="1793" alt="Screenshot 2026-02-24 154230" src="https://github.com/user-attachments/assets/08d764a8-2431-4245-8622-b773a4ddd9c9" />

## Abstract

> Effective hiring is integral to the success of an organisation, but it is very challenging to find the most suitable candidates because expert evaluation (e.g. interviews conducted by a technical manager) are expensive to deploy at scale. Therefore, automated resume scoring and other applicant-screening methods are increasingly used to coarsely filter candidates, making decisions on limited information. We propose that large language models (LLMs) can play the role of subject matter experts to cost-effectively elicit information from each candidate that is nuanced and role-specific, thereby improving the quality of early-stage hiring decisions. We present a system that leverages an LLM interviewer to update belief over an applicant's rubric-oriented latent traits in a calibrated way. We evaluate our system on simulated interviews and show that belief converges towards the simulated applicants' artificially-constructed latent ability levels. We release code, a modest dataset of public-domain/anonymised resumes, belief calibration tests, and simulated interviews, at [https://github.com/mbzuai-nlp/beyond-the-resume](https://github.com/mbzuai-nlp/beyond-the-resume). Our demo is available at [https://btr.hstu.net](https://btr.hstu.net).

## Demo

[Demo](https://btr.hstu.net)

[Video](https://vimeo.com/1169118079?share=copy&fl=sv&fe=ci)

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

## Reproducing our results

This repository uses [DVC](https://dvc.org/) to define data pipelines. 

To run judge calibration tests:

```
uv run dvc exp run run-judge-tests
```

To run simulations:

```
uv run dvc exp run run-interview-simulation
```

## Running tests

```
uv run pytest
```