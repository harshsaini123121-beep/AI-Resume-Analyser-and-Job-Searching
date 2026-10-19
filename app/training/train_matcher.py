"""
Model Fine-Tuning Pipeline for Resume-Job Semantic Matching.
Fine-tunes Sentence-Transformers on domain-specific positive and negative resume-job pairs.
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESUMES_DIR = BASE_DIR / "data" / "training_resumes"
SAMPLE_RESUMES_DIR = BASE_DIR / "data" / "sample_resumes"
JOBS_CSV_PATH = BASE_DIR / "data" / "jobs.csv"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "fine_tuned_matcher"
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_training_dataset():
    """
    Constructs high-quality positive and negative (Resume, Job Description) training pairs
    from the expanded resumes corpus and jobs dataset.
    """
    df_jobs = pd.read_csv(JOBS_CSV_PATH)

    # Collect resumes
    resumes = []
    for p in list(RESUMES_DIR.glob("*.txt")) + list(SAMPLE_RESUMES_DIR.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        resumes.append({
            "filename": p.stem,
            "text": text[:2000]
        })

    print(f"[Training Data] Loaded {len(resumes)} resume profiles.")
    print(f"[Training Data] Loaded {len(df_jobs)} job postings.")

    training_pairs = []

    for res in resumes:
        name_lower = res["filename"].lower()
        res_text = res["text"]

        # 1. Positive matches (label = 0.85 to 1.0)
        if "ml" in name_lower or "ai" in name_lower:
            matched_jobs = df_jobs[df_jobs["title"].str.contains("Machine Learning|AI|Data Scientist|NLP|Deep Learning", case=False, na=False)]
            for _, j in matched_jobs.head(8).iterrows():
                jd_text = f"{j['title']} at {j['company']}. {j['description'][:1500]}"
                training_pairs.append((res_text, jd_text, 0.95))

            # Negatives (label = 0.05)
            neg_jobs = df_jobs[df_jobs["title"].str.contains("Frontend|Designer|Sales|iOS|Angular", case=False, na=False)]
            for _, j in neg_jobs.head(6).iterrows():
                jd_text = f"{j['title']} at {j['company']}. {j['description'][:1500]}"
                training_pairs.append((res_text, jd_text, 0.10))

        elif "fullstack" in name_lower or "web" in name_lower:
            matched_jobs = df_jobs[df_jobs["title"].str.contains("Full Stack|Frontend|React|TypeScript|Backend", case=False, na=False)]
            for _, j in matched_jobs.head(8).iterrows():
                jd_text = f"{j['title']} at {j['company']}. {j['description'][:1500]}"
                training_pairs.append((res_text, jd_text, 0.95))

            neg_jobs = df_jobs[df_jobs["title"].str.contains("Cybersecurity|Audio|Penetration|Hardware", case=False, na=False)]
            for _, j in neg_jobs.head(6).iterrows():
                jd_text = f"{j['title']} at {j['company']}. {j['description'][:1500]}"
                training_pairs.append((res_text, jd_text, 0.10))

        elif "devops" in name_lower or "cloud" in name_lower:
            matched_jobs = df_jobs[df_jobs["title"].str.contains("DevOps|Cloud|Kubernetes|SRE|Infrastructure", case=False, na=False)]
            for _, j in matched_jobs.head(8).iterrows():
                jd_text = f"{j['title']} at {j['company']}. {j['description'][:1500]}"
                training_pairs.append((res_text, jd_text, 0.95))

            neg_jobs = df_jobs[df_jobs["title"].str.contains("UI/UX|Flutter|Data Analyst|Design", case=False, na=False)]
            for _, j in neg_jobs.head(6).iterrows():
                jd_text = f"{j['title']} at {j['company']}. {j['description'][:1500]}"
                training_pairs.append((res_text, jd_text, 0.10))

        elif "data_analyst" in name_lower or "data_eng" in name_lower:
            matched_jobs = df_jobs[df_jobs["title"].str.contains("Data Analyst|Data Engineer|BI|Power BI|Snowflake|Analytics", case=False, na=False)]
            for _, j in matched_jobs.head(8).iterrows():
                jd_text = f"{j['title']} at {j['company']}. {j['description'][:1500]}"
                training_pairs.append((res_text, jd_text, 0.95))

            neg_jobs = df_jobs[df_jobs["title"].str.contains("Rust|Golang|Embedded|iOS|Security", case=False, na=False)]
            for _, j in neg_jobs.head(6).iterrows():
                jd_text = f"{j['title']} at {j['company']}. {j['description'][:1500]}"
                training_pairs.append((res_text, jd_text, 0.10))

        elif "cybersecurity" in name_lower:
            matched_jobs = df_jobs[df_jobs["title"].str.contains("Security|Penetration|Cyber|OWASP|Infosec", case=False, na=False)]
            for _, j in matched_jobs.head(8).iterrows():
                jd_text = f"{j['title']} at {j['company']}. {j['description'][:1500]}"
                training_pairs.append((res_text, jd_text, 0.95))

            neg_jobs = df_jobs[df_jobs["title"].str.contains("Data Scientist|Frontend|Tableau|React", case=False, na=False)]
            for _, j in neg_jobs.head(6).iterrows():
                jd_text = f"{j['title']} at {j['company']}. {j['description'][:1500]}"
                training_pairs.append((res_text, jd_text, 0.10))

    print(f"[Training Data] Formatted {len(training_pairs)} labeled resume-job alignment pairs.")
    return training_pairs


def train_model(epochs: int = 2, batch_size: int = 8):
    """
    Executes fine-tuning with Sentence-Transformers using CosineSimilarityLoss.
    Saves the trained model to models/fine_tuned_matcher.
    """
    print("==================================================")
    print("Starting Model Training & Fine-Tuning Pipeline")
    print("==================================================")

    training_pairs = build_training_dataset()
    if not training_pairs:
        print("No training pairs found. Ensure data/jobs.csv and resumes are populated.")
        return

    try:
        import torch
        import torch.nn as nn
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        base_model_name = "all-MiniLM-L6-v2"
        print(f"[Model Training] Loading base model '{base_model_name}'...")
        model = SentenceTransformer(base_model_name)

        examples = [
            InputExample(texts=[t1, t2], label=float(lbl))
            for t1, t2, lbl in training_pairs
        ]

        train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(model)

        print(f"[Model Training] Training for {epochs} epochs over {len(examples)} examples (batch size: {batch_size})...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=10,
            show_progress_bar=True
        )

        print(f"[Model Training] Saving fine-tuned checkpoint to: {OUTPUT_MODEL_DIR}")
        model.save(str(OUTPUT_MODEL_DIR))

        # Save metadata info
        info = {
            "base_model": base_model_name,
            "training_examples": len(examples),
            "epochs": epochs,
            "loss_function": "CosineSimilarityLoss",
            "status": "fine_tuned_successfully"
        }
        (OUTPUT_MODEL_DIR / "training_meta.json").write_text(json.dumps(info, indent=2))
        print("[Model Training] Training complete! Custom domain model saved successfully.")

    except Exception as e:
        print(f"[Model Training Notice] Sentence-transformers training deferred or offline: {e}")
        # Save manifest record
        info = {
            "dataset_pairs_generated": len(training_pairs),
            "status": "training_dataset_ready",
            "message": str(e)
        }
        (OUTPUT_MODEL_DIR / "training_meta.json").write_text(json.dumps(info, indent=2))
        print("[Model Training] Training data manifests generated successfully.")

    print("==================================================")


if __name__ == "__main__":
    train_model(epochs=2, batch_size=8)
