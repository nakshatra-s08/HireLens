"""
Taxonomy and Skill Dictionary for HireLens Explainable Recruiter.
Provides canonical skill names, aliases, categories, and regex pattern compilation.
"""

from typing import Dict, List, Set
import re

SKILL_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "Programming Languages": {
        "Python": ["python", "py", "python3"],
        "JavaScript": ["javascript", "js", "ecmascript"],
        "TypeScript": ["typescript", "ts"],
        "Java": ["java", "j2ee"],
        "C++": ["c++", "cpp"],
        "C#": ["c#", "csharp", ".net"],
        "Go": ["go", "golang"],
        "Rust": ["rust"],
        "Ruby": ["ruby"],
        "PHP": ["php"],
        "Swift": ["swift"],
        "Kotlin": ["kotlin"],
        "Scala": ["scala"],
        "R": ["r language", "r-lang", "r programming"],
        "SQL": ["sql", "tsql", "plsql"],
        "HTML/CSS": ["html", "css", "html5", "css3"],
        "Bash/Shell": ["bash", "shell", "sh", "zsh", "powershell"],
    },
    "Frameworks & Libraries": {
        "React": ["react", "reactjs", "react.js"],
        "Angular": ["angular", "angularjs"],
        "Vue.js": ["vue", "vuejs", "vue.js"],
        "Next.js": ["next.js", "nextjs"],
        "Node.js": ["node", "nodejs", "node.js"],
        "Express": ["express", "expressjs"],
        "Django": ["django"],
        "Flask": ["flask"],
        "FastAPI": ["fastapi"],
        "Spring Boot": ["spring", "springboot", "spring boot"],
        "PyTorch": ["pytorch", "torch"],
        "TensorFlow": ["tensorflow", "tf"],
        "Scikit-Learn": ["scikit-learn", "sklearn"],
        "Pandas": ["pandas"],
        "NumPy": ["numpy"],
        "OpenCV": ["opencv"],
        "Keras": ["keras"],
        "HuggingFace": ["huggingface", "transformers"],
        "Tailwind CSS": ["tailwind", "tailwindcss"],
        "Bootstrap": ["bootstrap"],
        "GraphQL": ["graphql"],
        "REST API": ["rest", "restful", "rest api", "web apis"],
    },
    "Cloud & DevOps": {
        "AWS": ["aws", "amazon web services", "s3", "ec2", "lambda"],
        "Azure": ["azure", "microsoft azure"],
        "GCP": ["gcp", "google cloud", "google cloud platform"],
        "Docker": ["docker", "containerization"],
        "Kubernetes": ["kubernetes", "k8s"],
        "Terraform": ["terraform"],
        "Ansible": ["ansible"],
        "Jenkins": ["jenkins"],
        "GitHub Actions": ["github actions", "gh actions"],
        "CI/CD": ["ci/cd", "cicd", "continuous integration"],
        "Linux": ["linux", "ubuntu", "debian", "centos", "rhel"],
        "Nginx": ["nginx"],
        "Helm": ["helm"],
        "Prometheus": ["prometheus"],
        "Grafana": ["grafana"],
    },
    "Databases": {
        "PostgreSQL": ["postgresql", "postgres", "pg"],
        "MySQL": ["mysql"],
        "MongoDB": ["mongodb", "mongo"],
        "Redis": ["redis"],
        "Elasticsearch": ["elasticsearch", "elastic"],
        "DynamoDB": ["dynamodb"],
        "Snowflake": ["snowflake"],
        "BigQuery": ["bigquery"],
        "Cassandra": ["cassandra"],
        "SQLite": ["sqlite"],
        "Neo4j": ["neo4j"],
    },
    "Data & AI / ML": {
        "Machine Learning": ["machine learning", "ml"],
        "Deep Learning": ["deep learning", "dl"],
        "NLP": ["nlp", "natural language processing"],
        "Computer Vision": ["computer vision", "cv"],
        "LLMs": ["llm", "llms", "large language models", "rag", "langchain"],
        "Apache Spark": ["spark", "pyspark", "apache spark"],
        "Apache Kafka": ["kafka", "apache kafka"],
        "Airflow": ["airflow", "apache airflow"],
        "Tableau": ["tableau"],
        "PowerBI": ["powerbi", "power bi"],
        "Data Engineering": ["data engineering", "etl", "data pipelines"],
    },
    "Tools & Practices": {
        "Git": ["git", "version control"],
        "GitHub": ["github"],
        "GitLab": ["gitlab"],
        "Jira": ["jira"],
        "Postman": ["postman"],
        "Figma": ["figma"],
        "Unit Testing": ["pytest", "jest", "unit testing", "junit", "testing"],
        "Microservices": ["microservices", "distributed systems"],
    },
    "Soft Skills & Management": {
        "Communication": ["communication", "verbal communication", "written communication"],
        "Problem Solving": ["problem solving", "analytical skills", "troubleshooting"],
        "Leadership": ["leadership", "team lead", "tech lead", "mentorship"],
        "Agile/Scrum": ["agile", "scrum", "kanban"],
        "Project Management": ["project management", "stakeholder management"],
        "Collaboration": ["collaboration", "cross-functional", "team player"],
    }
}

def get_alias_mapping() -> Dict[str, str]:
    """Map every lowercased alias to its canonical skill name."""
    mapping = {}
    for category, skills in SKILL_TAXONOMY.items():
        for canonical, aliases in skills.items():
            mapping[canonical.lower()] = canonical
            for alias in aliases:
                mapping[alias.lower()] = canonical
    return mapping

ALIAS_TO_CANONICAL = get_alias_mapping()

def get_canonical_to_category() -> Dict[str, str]:
    """Map canonical skill name to its category."""
    mapping = {}
    for category, skills in SKILL_TAXONOMY.items():
        for canonical in skills.keys():
            mapping[canonical] = category
    return mapping

CANONICAL_TO_CATEGORY = get_canonical_to_category()
