# Ersilia's analysis template

This repository provides a structured template for setting up new research analysis in Ersilia.

## Background

<Replace this paragraph with a short description of the project. This description should explain the background or context of the project, specifying collaborators.>

## 🚀 Getting Started

<Replace this bit with any relevant information about how to use this repository>

```bash
git clone <your-repo-url>
cd <your-repo-url>
```

### Tracking details

The project is is tracked in [GitHub](https://github.com/ersilia-os/) (code) and [EOSVC](https://github.com/ersilia-os/eosvc) (data):

* Tracked by Git and linked to a Github repository: only src, scripts and notebooks.
* Tracked by eosvc and linked to a public or private S3 bucket. Only the data/ and output/ folder are eosvc tracked.

## Repository structure

This repository is organized as follows:

```
eos-analysis-template/
│
├── LICENSE
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
├── notebooks/
├── assets/
├── output/
├── src/
├── tools/
├── docs/
├── tmp/
└── .git/
```

- **data/**
  - **raw/** → Original, untouched datasets  
  - **processed/** → Cleaned and transformed datasets  

- **scripts/** → Standalone scripts for preprocessing or automation. Numbered in sequential order for running 

- **notebooks/** → Jupyter notebooks for exploration and prototyping  

- **assets/** → Images, figures, and other static resources  

- **output/** → Results of the scripts, numbered by file or folder according to the scripts numbering  

- **src/** → Core source code and reusable modules  

- **tools/** → Helper utilities and development tools  

- **docs/** → Project documentation and reports, including AI-generated docs and files

- **tmp/** → Temporary files or intermediate outputs  

- **.git/** → Git metadata (version control)  

- **requirements.txt** → version-specified list of packages required to run the analysis

---

📌 Empty folders are preserved with `.gitkeep` files so the structure remains consistent in Git.

---

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
