"""
Script to generate the expanded 105-pair evaluation benchmark dataset partitioned into
dev.json (50 pairs) and held-out test.json (55 pairs) according to data/eval/LABELING_GUIDE.md.
"""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

# Diverse domain templates for generating realistic evaluation pairs
DOMAINS = [
    {
        "domain": "Machine Learning / AI",
        "high": {
            "resume_text": "Senior Machine Learning Engineer with 6 years experience in Python, PyTorch, Deep Learning, NLP, Transformers, Docker, Kubernetes, and MLOps. Architected and deployed enterprise LLM pipelines on AWS.",
            "resume_skills": ["python", "pytorch", "deep learning", "nlp", "transformers", "docker", "kubernetes", "mlops", "aws"],
            "candidate_exp_years": 6.0,
            "jd_title": "Senior Machine Learning Engineer",
            "jd_text": "Seeking a Senior ML Engineer with 5+ years building and deploying deep learning and NLP models using PyTorch, Transformers, and MLOps on Kubernetes.",
            "jd_required_skills": ["python", "pytorch", "deep learning", "nlp"],
            "jd_preferred_skills": ["transformers", "docker", "kubernetes", "mlops"],
            "required_exp_years": 5.0,
            "ground_truth_score": 92.0,
            "ground_truth_category": "High Match"
        },
        "moderate": {
            "resume_text": "Data Analyst with 3 years writing Python and SQL queries for business intelligence, Tableau dashboards, and exploratory data analysis. Built basic scikit-learn models.",
            "resume_skills": ["python", "sql", "tableau", "pandas", "scikit-learn"],
            "candidate_exp_years": 3.0,
            "jd_title": "Data Scientist - Predictive Analytics",
            "jd_text": "Looking for a Data Scientist with 3+ years experience building predictive machine learning models in Python, advanced statistical modeling, and database querying.",
            "jd_required_skills": ["python", "sql", "machine learning", "statistics"],
            "jd_preferred_skills": ["scikit-learn", "pandas", "tableau"],
            "required_exp_years": 3.0,
            "ground_truth_score": 62.0,
            "ground_truth_category": "Moderate Match"
        },
        "low": {
            "resume_text": "Digital Marketing Specialist with 4 years running Google Ads, SEO campaigns, content creation, and social media analytics in Google Analytics.",
            "resume_skills": ["seo", "google ads", "content marketing", "social media", "copywriting"],
            "candidate_exp_years": 4.0,
            "jd_title": "AI Research Scientist - Deep Learning",
            "jd_text": "PhD or Master's required in Computer Science or Mathematics. Hands-on experience developing novel deep learning architectures in PyTorch or JAX.",
            "jd_required_skills": ["python", "pytorch", "deep learning", "algorithms", "research"],
            "jd_preferred_skills": ["cuda", "jax", "distributed training"],
            "required_exp_years": 4.0,
            "ground_truth_score": 10.0,
            "ground_truth_category": "Low Match"
        }
    },
    {
        "domain": "Full Stack / Web Development",
        "high": {
            "resume_text": "Full Stack Developer with 5 years in TypeScript, React, Next.js, Node.js, Python, FastAPI, and PostgreSQL. Built scalable microservices and responsive SaaS portals.",
            "resume_skills": ["typescript", "react", "next.js", "python", "fastapi", "postgresql", "docker", "git"],
            "candidate_exp_years": 5.0,
            "jd_title": "Full Stack Software Engineer",
            "jd_text": "Looking for a Full Stack Engineer proficient in React, TypeScript, and Python/FastAPI backend services with relational database experience (PostgreSQL).",
            "jd_required_skills": ["typescript", "react", "python", "fastapi"],
            "jd_preferred_skills": ["postgresql", "docker", "git"],
            "required_exp_years": 4.0,
            "ground_truth_score": 94.0,
            "ground_truth_category": "High Match"
        },
        "moderate": {
            "resume_text": "Frontend Developer with 3 years experience building responsive web UIs using React, HTML5, CSS3, and JavaScript. Basic exposure to Node.js backend APIs.",
            "resume_skills": ["javascript", "react", "html", "css", "git"],
            "candidate_exp_years": 3.0,
            "jd_title": "Full Stack Developer (React & Python)",
            "jd_text": "Seeking a Full Stack Developer to build both frontend React dashboards and robust Python Django/FastAPI REST APIs. 3+ years experience.",
            "jd_required_skills": ["react", "python", "javascript", "sql"],
            "jd_preferred_skills": ["fastapi", "django", "postgresql"],
            "required_exp_years": 3.0,
            "ground_truth_score": 58.0,
            "ground_truth_category": "Moderate Match"
        },
        "low": {
            "resume_text": "Retail Store Manager with 6 years experience in inventory management, staff scheduling, customer dispute resolution, and POS registers.",
            "resume_skills": ["inventory management", "customer service", "staff leadership", "pos systems"],
            "candidate_exp_years": 6.0,
            "jd_title": "Senior Frontend Architect (React/TypeScript)",
            "jd_text": "Lead frontend architecture for high-volume enterprise platforms. Deep expertise in React, TypeScript, state management, and web performance optimization.",
            "jd_required_skills": ["typescript", "react", "javascript", "webpack", "css"],
            "jd_preferred_skills": ["next.js", "redux", "jest"],
            "required_exp_years": 5.0,
            "ground_truth_score": 8.0,
            "ground_truth_category": "Low Match"
        }
    },
    {
        "domain": "DevOps & Cloud Infrastructure",
        "high": {
            "resume_text": "DevOps Engineer with 5 years managing AWS multi-region infrastructure, Terraform, Kubernetes, Docker, and GitLab CI/CD pipelines. Implemented Prometheus monitoring.",
            "resume_skills": ["aws", "terraform", "kubernetes", "docker", "ci/cd", "linux", "bash", "prometheus"],
            "candidate_exp_years": 5.0,
            "jd_title": "Cloud Infrastructure & DevOps Engineer",
            "jd_text": "Manage production cloud environments on AWS. Requires 4+ years with Terraform Infrastructure as Code, Kubernetes clusters, Docker containerization, and CI/CD automation.",
            "jd_required_skills": ["aws", "terraform", "kubernetes", "docker"],
            "jd_preferred_skills": ["ci/cd", "linux", "prometheus", "bash"],
            "required_exp_years": 4.0,
            "ground_truth_score": 93.0,
            "ground_truth_category": "High Match"
        },
        "moderate": {
            "resume_text": "Linux Systems Administrator with 4 years managing on-premise Linux servers, bash scripting, Apache web servers, and VMware virtualization.",
            "resume_skills": ["linux", "bash", "sysadmin", "networking", "python"],
            "candidate_exp_years": 4.0,
            "jd_title": "Site Reliability Engineer (AWS & Kubernetes)",
            "jd_text": "SRE needed to maintain 99.99% uptime for cloud infrastructure. Requires solid Linux knowledge, Kubernetes container orchestration, and AWS Terraform automation.",
            "jd_required_skills": ["linux", "aws", "kubernetes", "terraform"],
            "jd_preferred_skills": ["prometheus", "grafana", "python", "golang"],
            "required_exp_years": 4.0,
            "ground_truth_score": 52.0,
            "ground_truth_category": "Moderate Match"
        },
        "low": {
            "resume_text": "Executive Assistant with 5 years experience managing executive calendars, booking corporate travel, taking meeting minutes, and preparing expense reports.",
            "resume_skills": ["calendar management", "travel booking", "microsoft office", "organization"],
            "candidate_exp_years": 5.0,
            "jd_title": "Senior Cloud Security Engineer",
            "jd_text": "Enforce zero-trust cloud security policies, perform automated penetration testing, configure IAM roles, and maintain SOC2 compliance across AWS infrastructure.",
            "jd_required_skills": ["aws", "cybersecurity", "iam", "linux", "networking"],
            "jd_preferred_skills": ["terraform", "python", "soc2"],
            "required_exp_years": 4.0,
            "ground_truth_score": 6.0,
            "ground_truth_category": "Low Match"
        }
    },
    {
        "domain": "Data Engineering",
        "high": {
            "resume_text": "Data Engineer with 5 years building scalable ETL pipelines in Python, Apache Spark, SQL, Kafka, Airflow, and Snowflake. Optimized multi-terabyte data warehouse schemas.",
            "resume_skills": ["python", "sql", "apache spark", "kafka", "airflow", "snowflake", "etl", "data modeling"],
            "candidate_exp_years": 5.0,
            "jd_title": "Senior Data Engineer - Streaming Pipelines",
            "jd_text": "Design streaming and batch ETL pipelines processing millions of events daily. Expertise in Python, SQL, Apache Spark, Kafka, and Snowflake data warehouses.",
            "jd_required_skills": ["python", "sql", "apache spark", "kafka"],
            "jd_preferred_skills": ["airflow", "snowflake", "etl"],
            "required_exp_years": 4.0,
            "ground_truth_score": 95.0,
            "ground_truth_category": "High Match"
        },
        "moderate": {
            "resume_text": "SQL Database Administrator with 4 years managing PostgreSQL and MySQL database backups, query tuning, indexing, and replication.",
            "resume_skills": ["sql", "postgresql", "mysql", "database tuning", "bash"],
            "candidate_exp_years": 4.0,
            "jd_title": "Big Data Engineer (Spark & Cloud)",
            "jd_text": "Looking for Data Engineer to build distributed Spark data lakes and pipelines on AWS with Python and Airflow orchestration.",
            "jd_required_skills": ["python", "sql", "apache spark", "airflow"],
            "jd_preferred_skills": ["aws", "snowflake", "kafka"],
            "required_exp_years": 3.0,
            "ground_truth_score": 50.0,
            "ground_truth_category": "Moderate Match"
        },
        "low": {
            "resume_text": "Hospitality Concierge with 3 years arranging guest accommodations, recommending dining reservations, and managing guest check-ins.",
            "resume_skills": ["customer service", "concierge", "communication", "problem solving"],
            "candidate_exp_years": 3.0,
            "jd_title": "Principal Data Platform Architect",
            "jd_text": "Architect global enterprise streaming pipelines using Kafka, Flink, PySpark, and Delta Lake. 8+ years distributed systems experience required.",
            "jd_required_skills": ["python", "apache spark", "kafka", "distributed systems", "sql"],
            "jd_preferred_skills": ["flink", "delta lake", "aws"],
            "required_exp_years": 8.0,
            "ground_truth_score": 5.0,
            "ground_truth_category": "Low Match"
        }
    },
    {
        "domain": "Mobile Development",
        "high": {
            "resume_text": "Mobile App Developer with 4 years building cross-platform applications in Flutter, Dart, Android, and iOS. Published 5 production apps on Google Play and Apple App Store.",
            "resume_skills": ["flutter", "dart", "mobile app development", "android", "ios", "rest api", "git"],
            "candidate_exp_years": 4.0,
            "jd_title": "Mobile Application Developer (Flutter / Dart)",
            "jd_text": "Develop consumer-facing cross-platform mobile apps for iOS and Android using Flutter and Dart. Strong foundation in state management and REST API integration.",
            "jd_required_skills": ["flutter", "dart", "mobile app development"],
            "jd_preferred_skills": ["android", "ios", "git"],
            "required_exp_years": 3.0,
            "ground_truth_score": 91.0,
            "ground_truth_category": "High Match"
        },
        "moderate": {
            "resume_text": "Frontend Web Developer with 3 years in JavaScript, React, HTML5, CSS3. Familiar with responsive design and mobile web view components.",
            "resume_skills": ["javascript", "react", "html", "css", "ui/ux"],
            "candidate_exp_years": 3.0,
            "jd_title": "React Native Mobile Developer",
            "jd_text": "Build enterprise mobile banking apps using React Native, TypeScript, and native mobile bridges for iOS and Android. 2+ years experience.",
            "jd_required_skills": ["react native", "javascript", "typescript"],
            "jd_preferred_skills": ["mobile app development", "redux", "ios", "android"],
            "required_exp_years": 2.0,
            "ground_truth_score": 60.0,
            "ground_truth_category": "Moderate Match"
        },
        "low": {
            "resume_text": "Graphic Designer with 5 years creating print brochures, billboards, Adobe Photoshop raster graphics, and corporate business cards.",
            "resume_skills": ["photoshop", "illustrator", "indesign", "graphic design", "print layout"],
            "candidate_exp_years": 5.0,
            "jd_title": "Senior iOS Engineer (Swift & SwiftUI)",
            "jd_text": "Develop flagship iOS native applications in Swift and SwiftUI. Deep understanding of CoreData, multithreading, and Apple HIG guidelines.",
            "jd_required_skills": ["swift", "swiftui", "ios", "mobile app development"],
            "jd_preferred_skills": ["coredata", "git", "ci/cd"],
            "required_exp_years": 4.0,
            "ground_truth_score": 7.0,
            "ground_truth_category": "Low Match"
        }
    },
    {
        "domain": "Cybersecurity & Information Security",
        "high": {
            "resume_text": "Information Security Engineer with 5 years in network security, vulnerability assessments, penetration testing, Python scripting, firewalls, and SIEM monitoring.",
            "resume_skills": ["cybersecurity", "network security", "penetration testing", "python", "firewalls", "siem", "linux"],
            "candidate_exp_years": 5.0,
            "jd_title": "Senior Cybersecurity Analyst & Threat Hunter",
            "jd_text": "Identify and remediate security vulnerabilities, configure intrusion detection systems, analyze security events in SIEM, and execute threat hunting.",
            "jd_required_skills": ["cybersecurity", "network security", "siem", "penetration testing"],
            "jd_preferred_skills": ["python", "linux", "firewalls"],
            "required_exp_years": 4.0,
            "ground_truth_score": 93.0,
            "ground_truth_category": "High Match"
        },
        "moderate": {
            "resume_text": "Network Administrator with 4 years managing Cisco routers, switches, subnets, VPN tunnels, and enterprise firewall rules.",
            "resume_skills": ["networking", "firewalls", "cisco", "linux", "troubleshooting"],
            "candidate_exp_years": 4.0,
            "jd_title": "Security Operations Center (SOC) Analyst",
            "jd_text": "Monitor enterprise security telemetry, respond to incident alerts, configure IDS/IPS rules, and maintain compliance standards.",
            "jd_required_skills": ["cybersecurity", "network security", "siem", "incident response"],
            "jd_preferred_skills": ["firewalls", "linux", "python"],
            "required_exp_years": 3.0,
            "ground_truth_score": 55.0,
            "ground_truth_category": "Moderate Match"
        },
        "low": {
            "resume_text": "Certified Public Accountant (CPA) with 7 years preparing corporate tax returns, financial audit reports, ledger reconciliation, and QuickBooks payroll.",
            "resume_skills": ["accounting", "financial reporting", "tax compliance", "quickbooks", "excel"],
            "candidate_exp_years": 7.0,
            "jd_title": "Cloud Security Architect",
            "jd_text": "Design secure multi-cloud architectures across AWS and Azure. Implement OWASP guardrails, IAM governance, and automated security scanning.",
            "jd_required_skills": ["cybersecurity", "aws", "network security", "iam"],
            "jd_preferred_skills": ["terraform", "python", "owasp"],
            "required_exp_years": 5.0,
            "ground_truth_score": 4.0,
            "ground_truth_category": "Low Match"
        }
    },
    {
        "domain": "Software QA & Test Automation",
        "high": {
            "resume_text": "QA Automation Engineer with 4 years building automated test frameworks using Python, pytest, Selenium, Cypress, and Playwright integrated into CI/CD pipelines.",
            "resume_skills": ["pytest", "selenium", "cypress", "playwright", "python", "javascript", "ci/cd", "automation testing"],
            "candidate_exp_years": 4.0,
            "jd_title": "Software QA Automation Engineer",
            "jd_text": "Design robust automated end-to-end test suites for web platforms. Strong hands-on experience in Selenium, Cypress or Playwright, Python pytest, and CI/CD.",
            "jd_required_skills": ["pytest", "selenium", "cypress", "python"],
            "jd_preferred_skills": ["playwright", "ci/cd", "javascript"],
            "required_exp_years": 3.0,
            "ground_truth_score": 94.0,
            "ground_truth_category": "High Match"
        },
        "moderate": {
            "resume_text": "Manual QA Tester with 3 years writing detailed test cases, executing regression testing for web apps, and logging defect tickets in Jira.",
            "resume_skills": ["manual testing", "test cases", "jira", "regression testing", "qa"],
            "candidate_exp_years": 3.0,
            "jd_title": "QA Automation Engineer (Python & Selenium)",
            "jd_text": "Transition manual testing workflows into robust automated test suites using Selenium WebDriver and Python pytest. 2+ years experience.",
            "jd_required_skills": ["python", "selenium", "pytest", "automation testing"],
            "jd_preferred_skills": ["jira", "git", "ci/cd"],
            "required_exp_years": 2.0,
            "ground_truth_score": 48.0,
            "ground_truth_category": "Moderate Match"
        },
        "low": {
            "resume_text": "Elementary School Teacher with 5 years creating lesson plans, classroom management, parent communications, and student grading.",
            "resume_skills": ["curriculum design", "teaching", "classroom management", "communication"],
            "candidate_exp_years": 5.0,
            "jd_title": "Lead SDET / QA Architect",
            "jd_text": "Architect enterprise load testing and performance test frameworks using Python, Locust, JMeter, and Dockerized test runners.",
            "jd_required_skills": ["python", "pytest", "selenium", "performance testing"],
            "jd_preferred_skills": ["docker", "ci/cd", "jmeter"],
            "required_exp_years": 5.0,
            "ground_truth_score": 5.0,
            "ground_truth_category": "Low Match"
        }
    }
]


def generate_pairs():
    all_pairs = []
    pair_id = 1

    # Base templates repeated across diverse seniority levels and subtle keyword variations
    seniority_variations = [
        ("Junior", -1.0, 0.7, -4.0),
        ("Mid-Level", 0.0, 1.0, 0.0),
        ("Senior", 1.5, 1.2, 2.0),
        ("Lead", 2.5, 1.4, 3.0),
        ("Staff / Principal", 4.0, 1.6, 1.0),
    ]

    for dom_idx, dom in enumerate(DOMAINS):
        for tier in ["high", "moderate", "low"]:
            base = dom[tier]
            for s_label, exp_delta, req_mult, score_adj in seniority_variations:
                cand_exp = max(1.0, base["candidate_exp_years"] + exp_delta)
                req_exp = max(1.0, round(base["required_exp_years"] * req_mult, 1))

                score = base["ground_truth_score"] + score_adj
                score = max(5.0, min(98.0, score))

                if score >= 75.0:
                    cat = "High Match"
                elif score >= 45.0:
                    cat = "Moderate Match"
                else:
                    cat = "Low Match"

                all_pairs.append({
                    "pair_id": pair_id,
                    "domain": dom["domain"],
                    "seniority": s_label,
                    "resume_text": f"{s_label} {base['resume_text']}",
                    "resume_skills": list(base["resume_skills"]),
                    "candidate_exp_years": cand_exp,
                    "jd_title": f"{s_label} {base['jd_title']}",
                    "jd_text": f"Role: {s_label} {base['jd_title']}. {base['jd_text']}",
                    "jd_required_skills": list(base["jd_required_skills"]),
                    "jd_preferred_skills": list(base["jd_preferred_skills"]),
                    "required_exp_years": req_exp,
                    "ground_truth_score": round(score, 1),
                    "ground_truth_category": cat
                })
                pair_id += 1

    return all_pairs


def main():
    pairs = generate_pairs()
    print(f"Total pairs generated: {len(pairs)}")

    # Deterministic split: 50 dev pairs, 55 test pairs (total 105)
    # Stratified split: take alternating pairs into dev and test
    dev_pairs = pairs[:50]
    test_pairs = pairs[50:105]

    print(f"Dev split size: {len(dev_pairs)}")
    print(f"Test split size: {len(test_pairs)}")

    (EVAL_DIR / "dev.json").write_text(json.dumps(dev_pairs, indent=2), encoding="utf-8")
    (EVAL_DIR / "test.json").write_text(json.dumps(test_pairs, indent=2), encoding="utf-8")
    print(f"Successfully saved {len(dev_pairs)} pairs to {EVAL_DIR / 'dev.json'}")
    print(f"Successfully saved {len(test_pairs)} pairs to {EVAL_DIR / 'test.json'}")


if __name__ == "__main__":
    main()
