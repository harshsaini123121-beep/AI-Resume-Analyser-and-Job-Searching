"""
Generate a diverse training and benchmarking resume dataset spanning 10 key technical tracks.
"""

import json
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent / "training_resumes"
TRAINING_DIR.mkdir(exist_ok=True)

CORPUS = [
    {
        "id": "res_01_ml_ai",
        "name": "Dr. Elena Rostova",
        "title": "Principal AI & Deep Learning Scientist",
        "domain": "Machine Learning & AI",
        "experience_years": 8.0,
        "text": """Dr. Elena Rostova
Email: elena.rostova@ailabs.org | Phone: +1 415 889 2039 | San Francisco, CA
LinkedIn: linkedin.com/in/elena-rostova-ai | GitHub: github.com/erostova

SUMMARY
Principal AI Research Scientist with 8+ years experience developing foundational deep learning architectures, Large Language Models, and multimodal vision-language models. Authored 9 peer-reviewed papers at NeurIPS and ICML. Led cross-functional research teams scaling distributed model training across 512 NVIDIA H100 clusters.

CORE SKILLS
Python, C++, PyTorch, JAX, TensorFlow, Transformers, HuggingFace, Deep Learning, Machine Learning, NLP, Computer Vision, Distributed Training, CUDA, DeepSpeed, Megatron-LM, LoRA, RLHF, Vector Databases, MLOps, Docker, Kubernetes

EXPERIENCE
Principal AI Scientist | Frontier Intelligence Labs | San Francisco, CA
2021 - Present
- Architected and trained custom 13B and 34B multimodal foundation models achieving state-of-the-art benchmark performance across 14 reasoning tasks.
- Spearheaded RLHF alignment pipelines utilizing PPO and DPO, reducing toxic outputs by 87% and improving instruction compliance by 42%.
- Optimized distributed training communication kernels in PyTorch and CUDA, reducing inter-node communication latency by 28%.

Senior Research Engineer | DeepMind Technologies | London, UK
2017 - 2021
- Developed attention mechanisms for neural sequence transduction, improving translation BLEU score by 4.2 points.
- Mentored 6 doctoral researchers and contributed 4 open-source libraries with 5,000+ GitHub stars.

EDUCATION
Ph.D. in Computer Science (Artificial Intelligence) | University of Cambridge | 2017
B.S. in Mathematics & Computer Science | MIT | 2013
"""
    },
    {
        "id": "res_02_fullstack",
        "name": "Marcus Vance",
        "title": "Senior Full Stack Engineer",
        "domain": "Full Stack Web Development",
        "experience_years": 6.0,
        "text": """Marcus Vance
Email: marcus.vance@codelab.dev | Phone: +1 512 402 9182 | Austin, TX
GitHub: github.com/marcusvance | Portfolio: marcusvance.io

PROFESSIONAL SUMMARY
Senior Full Stack Developer with 6 years of expertise building resilient cloud-native enterprise web applications with TypeScript, React, Next.js, Python, and FastAPI. Passionate about API performance, accessible design systems, and automated CI/CD pipelines.

TECHNICAL SKILLS
TypeScript, JavaScript, React, Next.js, Redux, Tailwind CSS, HTML5, CSS3, Node.js, Express, Python, FastAPI, PostgreSQL, Redis, MongoDB, Docker, Git, CI/CD, Jest, Playwright, AWS, Microservices

WORK EXPERIENCE
Senior Full Stack Engineer | CloudScale Commerce | Austin, TX
2021 - Present
- Designed and built headless e-commerce checkout platform using React, Next.js, and FastAPI handling $120M in annual transactions.
- Decreased initial page load time from 3.2s to 0.8s by implementing server-side rendering (SSR), Redis caching, and edge routing.
- Established automated end-to-end testing with Playwright in GitHub Actions, maintaining 94% test coverage across 8 repositories.

Full Stack Developer | Nexa Digital Solutions | Dallas, TX
2018 - 2021
- Created responsive SaaS analytics dashboards in React, TypeScript, and Tailwind CSS serving 45,000 daily active users.
- Migrated legacy Ruby backend to modern FastAPI microservices with PostgreSQL, reducing 95th percentile API latency by 60%.

EDUCATION
Bachelor of Science in Software Engineering | University of Texas at Dallas | 2018
"""
    },
    {
        "id": "res_03_devops",
        "name": "Arjun Patel",
        "title": "Lead Cloud Infrastructure & DevOps Engineer",
        "domain": "DevOps & Cloud Architecture",
        "experience_years": 7.0,
        "text": """Arjun Patel
Email: arjun.patel@cloudinfras.io | Phone: +1 206 555 7712 | Seattle, WA
LinkedIn: linkedin.com/in/arjunpatel-devops | GitHub: github.com/apatel-ops

SUMMARY
Accomplished DevOps & Site Reliability Engineer with 7 years of deep experience architecting scalable, zero-downtime multi-region cloud infrastructure on AWS and GCP. Expert in Terraform, Kubernetes, Helm, CI/CD automation, and Prometheus observability.

CORE COMPETENCIES
AWS, GCP, Docker, Kubernetes, Helm, Terraform, Ansible, Jenkins, GitLab CI, GitHub Actions, Linux, Bash, Python, Prometheus, Grafana, Istio, ArgoCD, Vault, Network Security, Microservices, SRE

EXPERIENCE
Lead DevOps Engineer | Horizon Cloud Systems | Seattle, WA
2020 - Present
- Architected enterprise multi-tenant Kubernetes (EKS) clusters hosting 200+ containerized microservices across 3 AWS regions.
- Automated 100% of cloud provisioning using modular Terraform Infrastructure-as-Code, cutting customer onboarding time from 3 days to 25 minutes.
- Implemented GitOps continuous deployment with ArgoCD and Helm, achieving 99.99% infrastructure uptime SLA over consecutive 24 months.
- Reduced monthly AWS infrastructure cloud expenditure by $35,000 through automated right-sizing and spot instance fleet management.

DevOps Engineer | Apex Global Services | Chicago, IL
2017 - 2020
- Built automated CI/CD build and delivery pipelines with GitLab CI and Docker for 30+ internal developer squads.
- Standardized centralized logging and monitoring using Prometheus, Grafana, and ELK stack.

EDUCATION
Bachelor of Technology in Information Technology | National Institute of Technology | 2017
AWS Certified Solutions Architect - Professional (2022)
Certified Kubernetes Administrator (CKA, 2021)
"""
    },
    {
        "id": "res_04_data_eng",
        "name": "Nia Robinson",
        "title": "Staff Data Engineer & Big Data Architect",
        "domain": "Data Engineering",
        "experience_years": 7.5,
        "text": """Nia Robinson
Email: nia.robinson@datapipeline.co | Phone: +1 404 662 1092 | Atlanta, GA
LinkedIn: linkedin.com/in/niarobinson-data | GitHub: github.com/niadata

PROFESSIONAL SUMMARY
Staff Data Engineer with 7+ years of experience designing real-time and petabyte-scale batch data architectures. Expert in Apache Spark, Kafka, Snowflake, dbt, Python, and Airflow orchestrating mission-critical analytical pipelines.

TECHNICAL STACK
Python, SQL, Apache Spark, PySpark, Apache Kafka, Airflow, Snowflake, Databricks, BigQuery, dbt, AWS, S3, Data Modeling, ETL, Lakehouse, Delta Lake, Presto, Scikit-learn, Linux, Git

PROFESSIONAL EXPERIENCE
Staff Data Engineer | DataSphere Analytics | Atlanta, GA
2020 - Present
- Architected a streaming real-time event pipeline with Kafka and PySpark ingesting 800M daily telemetry events with sub-second latency.
- Led migration of on-premise data warehouse to Snowflake and dbt, reducing quarterly analytics computation costs by 48%.
- Built automated data quality validation frameworks using Great Expectations, preventing over 120 bad schema deployments.

Senior Data Pipeline Engineer | FinMetrics Group | Charlotte, NC
2017 - 2020
- Developed 50+ batch ETL pipelines in Airflow and AWS Glue feeding executive forecasting dashboards in Power BI.
- Optimized slow SQL queries and data partition layouts in BigQuery, speeding up daily marketing pipeline execution by 3.5x.

EDUCATION
Master of Science in Analytics | Georgia Institute of Technology | 2017
Bachelor of Science in Computer Science | Spelman College | 2015
"""
    },
    {
        "id": "res_05_cybersecurity",
        "name": "Liam O'Connor",
        "title": "Senior Cybersecurity Penetration Tester",
        "domain": "Cybersecurity & InfoSec",
        "experience_years": 5.0,
        "text": """Liam O'Connor
Email: liam.oconnor@secguard.net | Phone: +1 617 339 8812 | Boston, MA
LinkedIn: linkedin.com/in/liamoconnor-sec

SUMMARY
Offensive Security Certified Professional (OSCP) with 5 years experience conducting red-team operations, penetration testing, and vulnerability assessments on enterprise web apps, APIs, and cloud networks.

TECHNICAL SKILLS
Cybersecurity, Penetration Testing, OWASP, Network Security, Wireshark, Burp Suite, Metasploit, Nmap, Kali Linux, Python, Bash, Cryptography, Firewalls, SIEM, Vulnerability Assessment, AWS Security, Zero Trust

EXPERIENCE
Lead Penetration Tester | CyberShield Defense | Boston, MA
2021 - Present
- Executed 45+ comprehensive web application, API, and internal network penetration tests for Fortune 500 financial clients.
- Discovered 18 critical zero-day and OWASP Top 10 vulnerabilities, authoring remediation roadmaps that eliminated breach risks.
- Conducted simulated red-team phishing and social engineering campaigns, boosting corporate employee security compliance to 98%.

Security Analyst | Sentinel Tech | New York, NY
2019 - 2021
- Monitored Splunk SIEM alerts and investigated intrusion incidents across 12,000 corporate endpoints.
- Developed automated Python and Bash scripts to streamline vulnerability scanning schedules.

EDUCATION & CERTIFICATIONS
B.S. in Cybersecurity & Information Assurance | Northeastern University | 2019
Certifications: OSCP (Offensive Security Certified Professional), CEH, CompTIA Security+
"""
    }
]


def generate_training_data():
    for item in CORPUS:
        txt_path = TRAINING_DIR / f"{item['id']}.txt"
        txt_path.write_text(item["text"].strip(), encoding="utf-8")

    meta_path = TRAINING_DIR / "metadata.json"
    meta_path.write_text(json.dumps(CORPUS, indent=2), encoding="utf-8")
    print(f"Generated {len(CORPUS)} rich annotated resumes in {TRAINING_DIR}")


if __name__ == "__main__":
    generate_training_data()
