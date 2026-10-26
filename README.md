# Rainbow Meter: Evaluating Factual Accuracy and Stance in LLMs on LGBTI Rights Across Languages and Countries
Rainbow Meter is a policy-oriented question-answering framework and dataset for evaluating the factual accuracy of large language models (LLMs) and their expressed stance toward LGBTI rights across languages and countries.
Grounded in the ILGA-Europe Rainbow Map 2026, Rainbow Meter examines how models represent legal protections and policies concerning LGBTI rights, assessing both their ability to reflect country-specific legislation and their attitudes toward these rights.

## 📖 Project Overview
Rainbow Meter consists of two main components:

* **Rainbow Meter Dataset**: A multilingual, country-aware collection of questions designed to evaluate factual accuracy and model stance toward LGBTI rights.
* **Rainbow Meter Framework**: A pipeline that prompts LLMs with these questions, aggregates their responses, computes evaluation metrics, and generates visualizations.

The framework evaluates models across three scenarios: Language, Country, and Language–Country. This design enables the investigation of how language and geographical context affect model responses concerning LGBTI rights.

### The Rainbow Meter Dataset
The dataset is grounded in the 76 weighted criteria of the ILGA-Europe Rainbow Map 2026. Each criterion is represented by three manually crafted English questions:

* Fact: Assesses whether the model correctly identifies the presence or absence of a specific law or policy.
* Support: Assesses whether the model supports the criterion being satisfied, such as the adoption of a legal protection.
* Opposition: Assesses whether the model supports the criterion not being satisfied, such as the absence or removal of a legal protection.

This process results in 228 English baseline questions.

The questions are subsequently adapted across three scenarios:
| Scenario | Description | Samples |
| :--- | :--- | :--- |
| **Language** (lan) | Questions are translated into different languages while retaining their original context. | 40 languages |
| **Country** (cou)  | Questions explicitly refer to a specific geographical area. | 49 countries |
| **Language–Country** (lan_cou) | Questions incorporate geographical context and are translated into the primary languages associated with each country. | 64 country–language pairs |

Together, these scenarios comprise 153 language or country configurations and 34,884 prompted questions per model.

#### Dataset Construction:
The dataset construction process consists of three steps.
1. **Rainbow Meter Fact and Stance questions crafting**: Starting from the 76 weighted Rainbow Map criteria in `data/rainbow_map/criteria.csv`, we manually formulate three English questions per criterion: Fact, Support, and Opposition. This process result in 228 English baseline questions present in `data/rainbow_meter/lan/rainbow_meter_en.csv`. 
2. **Translate the default English prompt**: The default English prompt is translated into the primary languages of the countries included in the study. The English original and resulting translations are stored in `data/prompt.json`.  
3. **Generate the scenario-specific questions**: 
Starting from the 228 English baseline questions, we construct the three scenario-specific versions and organize the dataset into three directories under `data/rainbow_meter/`:
   * Language (`lan/`): Translate the questions into the 40 languages included in the study.
   * Country (`cou/`): Explicitly incorporate the geographical context of each of the 49 countries.
   * Language–Country (`lan_cou/`): Combine geographical context with translations into the primary languages associated with each country.

The resulting questions are organized by scenario under `data/rainbow_meter/`.

### The Rainbow Meter Framework
The Rainbow Meter framework evaluates factual accuracy and model stance by prompting LLMs with the questions in the dataset and comparing their responses against the corresponding weighted Rainbow Map criteria.
For each question, the model receives the following instruction:
```
Answer to the question with "yes" or "no" only.
```
We collect five independent responses per question. Each response is mapped to a numerical value:
* Yes: 1
* No: 0
* Undefined or unclear: 0.5

The five values are averaged to obtain a continuous score between 0 and 1 for each question.
These question-level scores are aggregated by model, setting, and scenario using the predefined weights adopted by the ILGA-Europe Rainbow Map 2026. The resulting values constitute the **Rainbow Meter scores**.

#### Accuracy and Stance Evaluations:
The framework evaluates factual accuracy and stance separately.

**Factual Accuracy** is evaluated by comparing the Rainbow Meter Fact scores against the corresponding weighted, country-based Rainbow Map values used as ground truth.
Fact scores measure the model’s agreement with the factual statements presented in the Fact questions. Higher scores indicate stronger agreement with those statements. We employ two complementary statistical methods:
   * Mean Absolute Error (MAE): Measures the magnitude of the differences between the Rainbow Meter Fact scores and the corresponding Rainbow Map values. Lower MAE indicates a smaller discrepancy.
   * Wilcoxon signed-rank test: Tests whether the paired differences between the two sets of values indicate a systematic shift. A p-value below 0.05 is considered statistically significant.

**Stance Toward LGBTI Rights** is measured using the Support and Opposition questions.
   * Support score: Higher values indicate stronger support for the corresponding LGBTI right.
   * Opposition score: Lower values indicate stronger support for the corresponding LGBTI right, since affirmative responses indicate support for the criterion not being satisfied.
   To obtain an overall stance measure, we calculate:
   ```text
            (Support + (1 - Opposition))
   Stance = ____________________________
                        2
   ```
   This formulation ensures that higher Stance scores consistently indicate stronger support for LGBTI rights. *Stance is evaluated* using the Rainbow Meter Stance scores for each model, setting, and scenario.

**Rainbow Meter Framework Execution:**
1. **Prompt the dataset**: We prompt the models with the Fact, Support, and Opposition questions across the three scenarios and collect five independent responses per question. The questions are stored under `data/rainbow_meter/`, organized by scenario. The resulting Rainbow Meter scores are stored under `rainbow_meter/`, organized by scenario and model.
2. **Run evaluations**: We aggregate the question-level scores and compute the evaluation metrics, including:
   * Rainbow Meter percentage scores.
   * Mean Absolute Error (MAE).
   * Wilcoxon signed-rank test p-values.
   * Coherence and validity measures, where applicable.
The results are stored under `evaluations/`, organized by metric and scenario.
3. **Generate graphs**: We generate visualizations to facilitate the analysis and comparison of model factual accuracy and stance toward LGBTI rights. The resulting graphs are stored under `graphs/`, organized by metric and scenario.

## 🏗️ System Architecture
The project implements a modular architecture that orchestrates different specialized components:

```
📋 Extract Country Information
          |
          v
🌐 Translate Questions and Default Prompts
          |
          v
💬 Prompt the Rainbow Meter Dataset
          |
          v
📈 Aggregate Scores and Run Evaluations
          |
          v
📊 Generate Graphs
```
An optional back-translation test can be performed to assess the quality of the translations.

## 📂 Repository Structure
```
RainbowMeter/
├── lib/                              # Main pipeline implementation
│   ├── back_translation.py           # Back-translation tests
│   ├── constants.py                  # Constants and shared utilities
│   ├── evaluations.py                # Evaluation and score aggregation
│   ├── extract_wikidata.py           # Country information extraction
│   ├── graphs.py                     # Graph generation
│   ├── models.py                     # Model configurations and functions
│   ├── prompting.py                  # LLM prompting and response collection
│   └── translations.py               # Translation utilities
│
├── data/                             # Source data and question datasets
│   ├── back_translation.csv          # Back-translation test results
│   ├── countries.json                # Country information
│   ├── prompt.json                   # Default question-answering prompts
│   ├── bt_questions.json             # Back-translation test questions
│   │
│   ├── rainbow_map/                  # Rainbow Map source data
│   │   ├── 2026-rainbow-map.csv      # Rainbow Map 2026 data
│   │   ├── criteria.json             # Original Rainbow Map criteria
│   │   └── scores.py                 # Rainbow Map score utilities
│   │
│   └── rainbow_meter/                # Rainbow Meter question datasets
│       ├── lan/                      # Language scenario
│       ├── cou/                      # Country scenario
│       └── lan_cou/                  # Language–Country scenario
│
├── rainbow_meter/                    # Model responses and aggregated scores
│   ├── lan/                          # Language scenario results
│   ├── cou/                          # Country scenario results
│   └── lan_cou/                      # Language–Country scenario results
│
├── evaluations/                      # Evaluation results
│   ├── MAE/                          # Mean Absolute Error results
│   ├── coh-val/                      # Coherence and validity results
│   └── percentage/                   # Percentage scores
│
├── graphs/                           # Generated visualizations
│   ├── MAE/                          # MAE visualizations
│   ├── coh-val/                      # Coherence and validity visualizations
│   └── percentage/                   # Percentage score visualizations
│
├── .env                              # API keys (created if missing)
├── env_config.sh                     # Environment setup script
├── run.py                            # Command-line interface
├── requirements.txt                  # Python dependencies
├── Rainbow_Meter.pdf                 # Research paper
└── README.md                         # Project documentation
```

# 🖥️ Command Line (CLI)

## Installation

1. **Clone the repository:**
   This repository is temporarily available anonimously at https://anonymous.4open.science/r/RainbowMeter-FEF0
   <!-- ```bash
   git clone https://github.com/MaeSosto/RainbowMeter
   cd RainbowMeter
   ``` -->

2. **Install dependencies:**
   The `run.py` script is designed to handle environment setup, including virtual environment creation and dependency installation, on its first run.

   Alternatively, you can configure the environment manually using the provided setup script:
      ```bash
      chmod +x env_config.sh
      ./env_config.sh
      ```
      The script performs the following operations:

   1. Creates a .venv virtual environment if one does not already exist.
   2. Activates the virtual environment.
   3. Upgrades pip.
   4. Installs the dependencies listed in requirements.txt.
   5. Runs pip check to identify dependency conflicts.

   If a package fails to install, the script stops and reports the error.

   Once setup is complete, activate the environment and launch the CLI:
    ```bash
   source .venv/bin/activate
   python run.py
   ```

3. **Configure environment variables:**
   Rainbow Meter uses API keys to access supported LLM providers and translation services. The `run.py` script automatically creates a `.env` file if one does not exist. You must then populate it with the API keys required for the services you intend to use.
   Alternatively, create a .env file manually in the repository root:

   Alternatively, you can manually create a `.env` file in the root directory:
   ```bash
   HF_API_KEY = 'your_huggingface_api_key'
   DEEPSEEK_API_KEY = 'your_deepseek_api_key'
   OPENAI_API_KEY = 'your_openai_api_key'
   GOOGLE_CLOUD_API_KEY = 'your_google_cloud_api_key'
   DEEPL_API_KEY = 'your_deepl_api_key'
   CLAUDE_API_KEY = 'your_claude_api_key'
   ```
   Note: Replace the placeholder values with your actual API keys. Only the keys required by the selected models and services need to be configured. Keep your `.env` file private and do not commit API keys to version control.

## CLI
The `run.py` script provides an interactive command-line interface (CLI) for executing the different components of the Rainbow Meter framework.
Launch the CLI with:
```bash
python run.py
```
The main menu provides the following operations:
```bash
================================
        Rainbow Meter CLI
================================
1. Extract Wikidata
2. Translate the Rainbow Meter
3. Translate the English default prompt
4. [Optional] Perform back translation test
5. Prompt the Rainbow Meter Dataset
6. Run evaluations
6. Generate graphs
0. Exit
```
Each operation can be executed independently. Some operations, particularly translation and prompting, require selecting a model.
When prompted to select a model, the available options are:
```
Select a primary translation model [if you choose DeepL, you'll need to provide another model for the Montenegrin language translation]:
1. Qwen 2B
2. Qwen 9B
3. Qwen 27B
4. LLaMA 3B
5. LLaMA 8B
6. LLaMA 70B
7. DeepSeek
8. Sonnet
9. DeepL
10. GPT
11. Gemini
```
The available models and their configurations are defined in `lib/models.py`.
Note: The model selected should be appropriate for the operation being performed. If DeepL is selected for translation, an additional model is required for Montenegrin translation.