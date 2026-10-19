"""
Script to expand data/skills.json with extensive technology, data science,
cloud, cybersecurity, and engineering competencies from open taxonomies.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SKILLS_JSON_PATH = BASE_DIR / "data" / "skills.json"

# Extensive additional domain taxonomies
EXPANDED_TAXONOMY = {
    "programming": [
        "python", "java", "c++", "c", "c#", "javascript", "typescript", "ruby", "php", "swift",
        "kotlin", "go", "golang", "rust", "r", "scala", "dart", "shell", "bash", "powershell",
        "perl", "haskell", "lua", "matlab", "vba", "groovy", "julia", "elixir", "clojure", "assembly",
        "solidity", "fortran", "cobol", "erlang", "f#", "ocaml", "zig", "nim", "lisp", "prolog",
        "verilog", "vhdl", "racket", "actionscript", "delphi", "visual basic", "objective-c", "awk"
    ],
    "web_frontend": [
        "html", "html5", "css", "css3", "sass", "scss", "less", "bootstrap", "tailwind", "tailwindcss",
        "javascript", "typescript", "react", "react.js", "reactjs", "next.js", "nextjs", "vue", "vue.js",
        "vuejs", "angular", "angularjs", "svelte", "sveltekit", "redux", "zustand", "mobx", "jquery",
        "webpack", "vite", "webgl", "three.js", "responsive design", "figma", "storybook", "ui/ux",
        "dom", "ajax", "styled-components", "chakra ui", "material ui", "shadcn", "radix ui", "alpine.js",
        "astro", "remix", "lit", "stencil", "babel", "parcel", "rollup", "esbuild", "turbo", "turbopack",
        "webrtc", "wasm", "webassembly", "pwa", "progressive web app", "shadow dom", "css grid", "flexbox"
    ],
    "web_backend": [
        "node.js", "nodejs", "express", "express.js", "fastapi", "django", "flask", "spring", "spring boot",
        "asp.net", ".net core", "ruby on rails", "laravel", "nest.js", "nestjs", "graphql", "rest api",
        "restful api", "grpc", "microservices", "websockets", "celery", "jwt", "oauth", "api gateway",
        "gin", "fiber", "actix-web", "axum", "rocket", "tornado", "aiohttp", "bottle", "pyramid",
        "symphony", "cakephp", "codeigniter", "phoenix", "play framework", "dropwizard", "micronaut",
        "quarkus", "vert.x", "kong", "envoy", "traefik", "reverse proxy", "event driven architecture",
        "soap", "json-rpc", "trpc", "openresty", "haproxy", "socket.io", "kafka connect", "debezium"
    ],
    "data_engineering_analysis": [
        "sql", "nosql", "pandas", "numpy", "scipy", "power bi", "tableau", "excel", "advanced excel",
        "apache spark", "pyspark", "hadoop", "apache kafka", "kafka", "airflow", "apache airflow",
        "dbt", "databricks", "snowflake", "bigquery", "redshift", "data modeling", "etl", "elt",
        "data pipelines", "data warehousing", "data visualization", "looker", "business intelligence",
        "metabase", "qlik", "alteryx", "statistics", "eda", "data cleansing", "apache flink",
        "apache beam", "delta lake", "apache iceberg", "hudi", "trino", "presto", "clickhouse",
        "prefect", "dagster", "luigi", "talend", "informatica", "ssis", "ssas", "ssrs", "dax",
        "power query", "polars", "duckdb", "kinesis", "glue", "athena", "synapse analytics",
        "data lake", "data mesh", "lakehouse", "schema design", "star schema", "snowflake schema"
    ],
    "ml_ai_datascience": [
        "machine learning", "deep learning", "artificial intelligence", "nlp", "natural language processing",
        "computer vision", "opencv", "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras",
        "huggingface", "transformers", "bert", "gpt", "large language models", "llms", "genai",
        "generative ai", "langchain", "llamaindex", "rag", "retrieval augmented generation", "xgboost",
        "lightgbm", "catboost", "reinforcement learning", "spacy", "nltk", "gensim", "vector databases",
        "pinecone", "chromadb", "faiss", "weaviate", "qdrant", "milvus", "mlops", "mlflow", "wandb",
        "dvc", "neural networks", "cnn", "rnn", "lstm", "fine-tuning", "prompt engineering", "sentiment analysis",
        "yolo", "object detection", "image segmentation", "gan", "diffusion models", "stable diffusion",
        "whisper", "speech-to-text", "tts", "lora", "qlora", "tensorrt", "onnx", "triton inference server",
        "vllm", "tgi", "deepgram", "cohere", "anthropic", "openai", "gemini", "mistral", "ollama",
        "semantic search", "embeddings", "ranking", "recommender systems", "a/b testing", "causal inference"
    ],
    "cloud_devops": [
        "aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud platform",
        "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins", "gitlab ci", "github actions",
        "ci/cd", "continuous integration", "continuous deployment", "helm", "linux", "ubuntu", "nginx",
        "apache", "prometheus", "grafana", "cloudwatch", "serverless", "aws lambda", "cloudformation",
        "vagrant", "argo cd", "flux", "istio", "linkerd", "service mesh", "pulumi", "packer", "consul",
        "vault", "datadog", "new relic", "splunk", "elastic stack", "elk", "open telemetry", "jaeger",
        "terragrunt", "ecs", "eks", "aks", "gke", "fargate", "cloud run", "app engine", "s3", "ec2", "iam"
    ],
    "databases_caching": [
        "mysql", "postgresql", "postgres", "sqlite", "mongodb", "redis", "elasticsearch", "cassandra",
        "dynamodb", "mariadb", "oracle db", "sql server", "ms sql", "neo4j", "couchbase", "memcached",
        "firebase", "supabase", "database design", "indexing", "acid", "sharding", "replication",
        "cockroachdb", "tidb", "yugabyte", "aerospike", "scylladb", "hbase", "opensearch", "meilisearch",
        "typesense", "graphdb", "arangodb", "dgraph", "timescaledb", "influxdb", "questdb", "surrealdb"
    ],
    "mobile_development": [
        "android", "ios", "react native", "flutter", "swiftui", "xcode", "android studio",
        "mobile app development", "expo", "ionic", "cordova", "capactior", "jetpack compose",
        "objective-c", "swift", "kotlin multiplatform", "kmp", "cocoapods", "fastlane", "app store connect",
        "google play console", "realm", "coredata", "room database", "rxjava", "rxswift", "combine"
    ],
    "testing_qa": [
        "pytest", "unittest", "jest", "mocha", "selenium", "cypress", "playwright", "postman",
        "swagger", "jmeter", "tdd", "bdd", "unit testing", "integration testing", "automation testing",
        "smoke testing", "regression testing", "k6", "gatling", "locust", "cucumber", "jasmine",
        "karma", "robot framework", "appium", "testrail", "allure", "sonarqube", "load testing"
    ],
    "cybersecurity_networking": [
        "cybersecurity", "network security", "information security", "penetration testing", "owasp",
        "ssl/tls", "firewalls", "siem", "soc", "cryptography", "wireshark", "vulnerability assessment",
        "burp suite", "metasploit", "nmap", "kali linux", "zero trust", "iam", "saml", "oidc",
        "pki", "ddos protection", "waf", "cloudflare", "ids/ips", "snort", "suricata", "incident response"
    ],
    "management_methodologies": [
        "agile", "scrum", "kanban", "jira", "confluence", "git", "github", "gitlab", "bitbucket",
        "product management", "sdlc", "code review", "system design", "technical leadership",
        "scrum master", "sprint planning", "ci/cd pipelines", "okrs", "kpis", "waterfall", "prince2", "pmp"
    ],
    "soft_skills": [
        "communication", "teamwork", "leadership", "problem solving", "critical thinking",
        "adaptability", "time management", "collaboration", "work ethic", "attention to detail",
        "conflict resolution", "negotiation", "presentation skills", "mentoring", "ownership",
        "analytical thinking", "creative thinking", "interpersonal skills", "cross-functional",
        "stakeholder management", "active listening", "resilience", "decision making", "growth mindset"
    ]
}

EXPANDED_SYNONYMS = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "genai": "generative ai",
    "gen ai": "generative ai",
    "llm": "large language models",
    "llms": "large language models",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "torch": "pytorch",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "node": "node.js",
    "nodejs": "node.js",
    "react": "react.js",
    "reactjs": "react.js",
    "next": "next.js",
    "nextjs": "next.js",
    "vue": "vue.js",
    "vuejs": "vue.js",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "ms azure": "azure",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "docker container": "docker",
    "spark": "apache spark",
    "kafka": "apache kafka",
    "bi": "business intelligence",
    "powerbi": "power bi",
    "gql": "graphql",
    "rest": "rest api",
    "restful": "rest api",
    "rag": "retrieval augmented generation",
    "tdd": "test driven development",
    "prompting": "prompt engineering",
    "rdbms": "sql",
    "git version control": "git",
    "golang": "go",
    "kmp": "kotlin multiplatform",
    "mui": "material ui",
    "tailwind": "tailwindcss",
    "bs": "bootstrap",
    "pg": "postgresql",
    "ms sql": "sql server",
    "mssql": "sql server",
    "tf-idf": "nlp",
    "s-bert": "sentence-transformers",
    "bert": "transformers"
}


def expand_skills_file():
    with open(SKILLS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_cats = data.get("categories", {})
    existing_synonyms = data.get("synonyms", {})

    total_added = 0
    for cat_name, new_skills in EXPANDED_TAXONOMY.items():
        current_list = set(s.lower() for s in existing_cats.get(cat_name, []))
        for skill in new_skills:
            s_clean = skill.lower().strip()
            if s_clean not in current_list:
                current_list.add(s_clean)
                total_added += 1
        existing_cats[cat_name] = sorted(list(current_list))

    existing_synonyms.update(EXPANDED_SYNONYMS)

    data["categories"] = existing_cats
    data["synonyms"] = existing_synonyms

    total_skills_count = sum(len(lst) for lst in existing_cats.values())

    with open(SKILLS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Added {total_added} new skills.")
    print(f"Master skills repository now contains {total_skills_count} categorized skills across {len(existing_cats)} domains!")
    print(f"Synonym lookup dictionary contains {len(existing_synonyms)} aliases.")


if __name__ == "__main__":
    expand_skills_file()
