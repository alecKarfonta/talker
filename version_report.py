import importlib
import subprocess
import sys
import logging

def get_package_version(package):
    version = None
    try:
        version = importlib.import_module(package).__version__
    except (ImportError, AttributeError):
        logging.warning(f"Could not get version for package: {package}")
    
    if not version:
        try:
            result = subprocess.run(['pip', 'show', package], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split()[-1]
                    break
        except FileNotFoundError:
            logging.warning("pip not found. Unable to get package version.")
    
    return version

def get_cmake_version():
    try:
        result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        return result.stdout.split('\n')[0].split()[-1]
    except FileNotFoundError:
        return None

def get_ninja_version():
    try:
        result = subprocess.run(['ninja', '--version'], capture_output=True, text=True)
        return result.stdout.strip()
    except FileNotFoundError:
        return None

packages = [
  
    "psutil",
    "sentencepiece",
    "tqdm",
    "py-cpuinfo",
    "fastapi",
    "aiohttp",
    "openai",
    "uvicorn",
    "pydantic",
    "pillow",
    "prometheus_client",
    "prometheus_fastapi_instrumentator",
    "tiktoken",
    "lm_format_enforcer",
    "outlines",
    "typing_extensions",
    "filelock"
  
    "fuzzywuzzy",
    "flask",
    "Werkzeug",
    "pymilvus",
    "Flask-RESTful",
    "Flask-SQLAlchemy",
    "Flask-Marshmallow",
    "Marshmallow",
    "marshmallow-sqlalchemy",
    "psycopg2-binary",
    "flask-restplus",
    "flask_swagger",
    "flask_swagger_ui",
    "pyyaml",
    "minio",
    "pretty-errors",
    "datefinder",
    "numpy",
    "pandas",
    "torch",
    "transformers",
    "bitsandbytes",
    "sentence_transformers",
    "nltk",
    "spacy",
    "pdfminer-six",
    "pymupdf",
    "pyLDAvis",
    "ghostscript",
    "pdfminer-six"
]


print("Package Versions:")
print(f"cmake: {get_cmake_version()}")
print(f"ninja: {get_ninja_version()}")

requirements = []

for package in packages:
    version = get_package_version(package)
    if version:
        print(f"{package}: {version}")
        requirements.append(f"{package}=={version}")
    else:
        print(f"{package}: Not installed or version not available")

# Check Python version
python_version = sys.version.split()[0]
print(f"Python: {python_version}")

# Generate requirements.txt
with open('requirements.txt', 'w') as f:
    f.write(f"# Python {python_version}\n")
    for req in requirements:
        f.write(f"{req}\n")

print("\nGenerated requirements.txt with installed package versions.")
