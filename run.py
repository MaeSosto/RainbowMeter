import sys
import os
import subprocess
from pathlib import Path
import importlib.util

BASE_DIR = Path(__file__).resolve().parent

# --- Environment Management Functions ---

def parse_requirements():
    """Parse requirements.txt to get a list of required packages and their likely import names."""
    req_file_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.exists(req_file_path):
        # Fallback to hardcoded core list if requirements.txt is missing
        return {
            'langgraph': 'langgraph',
            'spacy': 'spacy',
            'sentence-transformers': 'sentence_transformers',
            'faiss-cpu': 'faiss',
            'rdflib': 'rdflib',
            'tqdm': 'tqdm'
        }
        
    requirements = {}
    with open(req_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Remove version specifiers (e.g., package==1.0.0 -> package)
            # Also handle <, >, >=, etc.
            import re
            app_name = re.split(r'[=<>!]', line)[0].strip()
            
            # Map PyPI package names to Import names
            # Most packages use underscores instead of hyphens (e.g. 'sentence-transformers' -> 'sentence_transformers')
            if 'faiss' in app_name: import_name = 'faiss'
            elif app_name == 'python-dotenv': import_name = 'dotenv'
            elif app_name == 'pyyaml': import_name = 'yaml'
            else: import_name = app_name.replace('-', '_')
            
            requirements[app_name] = import_name
            
    return requirements

def check_imports():
    """Check if dependencies from requirements.txt are installed."""
    print("Checking dependencies....")
    required_packages = parse_requirements()
    missing = []
    
    for pkg_name, import_name in required_packages.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pkg_name)
            print(f"- {pkg_name} is missing")
            #logger.info(f"\n- {pkg_name} is missing")
    return missing

import os
import sys
import subprocess


def manage_environment_setup():
    """
    Ensures the standard environment (.venv) exists and runs the
    application using its Python interpreter.

    If the current interpreter is not .venv, restart the script
    using .venv's Python interpreter.
    """

    # Always resolve paths relative to run.py, not the working directory.
    project_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(project_dir, ".venv")
    requirements_file = os.path.join(project_dir, "requirements.txt")

    # Determine the virtual environment's Python executable.
    if os.name == "nt":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    # 1. Create the virtual environment if it doesn't exist.
    if not os.path.exists(venv_python):
        print("🚀 Initializing standardized environment: '.venv'")

        try:
            print("   • Creating virtual environment...")
            subprocess.check_call([
                sys.executable, "-m", "venv", venv_dir
            ])

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)

    # 2. Check whether the current Python interpreter is inside the target venv.
    current_prefix = os.path.abspath(sys.prefix)
    target_prefix = os.path.abspath(venv_dir)

    if os.name == "nt":
        current_prefix = os.path.normcase(current_prefix)
        target_prefix = os.path.normcase(target_prefix)

    if current_prefix == target_prefix and sys.prefix != sys.base_prefix:
        print("✅ Running inside .venv.")
        return venv_python

    # 3. Install dependencies using .venv's Python.
    if not os.path.exists(requirements_file):
        print(f"❌ requirements.txt not found: {requirements_file}")
        sys.exit(1)

    try:
        print("📦 Installing dependencies into .venv...")

        subprocess.check_call([
            venv_python, "-m", "pip", "install",
            "-r", requirements_file
        ])

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

    # 4. Restart the script using .venv's Python.
    print("🔄 Restarting application using .venv...")
    os.execv(venv_python, [venv_python] + sys.argv)
    return venv_python

def check_env_setup():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

    api_keys = [
        'DEEPSEEK_API_KEY',
        'OPENAI_API_KEY',
        'GOOGLE_CLOUD_API_KEY',
        'HF_API_KEY',
        'DEEPL_API_KEY',
        'CLAUDE_API_KEY'
    ]

    # Create .env if it doesn't exist
    if not os.path.exists(env_path):
        try:
            with open(env_path, 'w', encoding='utf-8') as f:
                for key in api_keys:
                    f.write(f'{key}=""\n')

            print(f"📝 Created .env file: {env_path}")

        except OSError as e:
            print(f"❌ Failed to create .env file: {e}")
            return False

    # Read configured API keys
    configured_keys = {}

    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith('#') or '=' not in line:
                    continue

                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key in api_keys and value:
                    configured_keys[key] = True

    except OSError as e:
        print(f"❌ Error reading .env file: {e}")
        return False

    # Identify missing or empty API keys
    missing_keys = [
        key for key in api_keys
        if key not in configured_keys
    ]

    if missing_keys:
        print("\n" + "=" * 60)
        print("🔑 API KEY CONFIGURATION")
        print("=" * 60)
        print("The following API keys are missing or empty in .env:")
        for key in missing_keys:
            print(f"  - {key}")
        print(f"\nPlease edit the file:\n{env_path}")
        print("=" * 60 + "\n")
        return False

    print("✅ All API keys are configured.")
    return True


def select_model(deepl= True):
    from lib.constants import DEEPL, MODEL_LABEL
    if deepl:
        models = list(MODEL_LABEL.items())
    else:
        models = list(MODEL_LABEL.remove(DEEPL).items())
        
    for index, (_, label) in enumerate(models, start=1):
        print(f"{index}. {label}")

    while True:
        choice = input("\nEnter the model number: ").strip()

        try:
            choice = int(choice)
            if 1 <= choice <= len(models):
                model = models[choice - 1][0]
                print(f"\nSelected model: {MODEL_LABEL[model]}")
                return model
        except ValueError:
            pass

        print(f"Invalid choice. Enter a number between 1 and {len(models)}.")

if __name__ == "__main__":

    venv_python = manage_environment_setup()
    env = check_env_setup()
    
    #os.execv(venv_python, [venv_python] + sys.argv)
    while True:
        print("\n================================")
        print("       Rainbow Meter CLI")
        print("================================")
        print("1. Check environment and imports")
        print("2. Check API keys")
        print("3. Extract Wikidata")
        print("4. Translate the Rainbow Meter")
        print("5. Translate the English default prompt")
        print("6. [Optional] Perform back translation test")
        print("7. Prompt the Rainbow Meter Dataset")
        print("8. Run evaluations")
        print("9. Generate graphs")
        print("0. Exit")
        choice = input("\nSelect an operation: ").strip()
        
        if choice == "1":
            # This checks for imports, and if missing, offers to switch/create env
            new_python_exe = manage_environment_setup()
            #break
        
        if choice == "2":
            #Check the .env file with the API KEYs
            env = check_env_setup()
            #break
        
        #Extract Wikidata
        if choice == "3":
            script_path = BASE_DIR / "lib/extract_wikidata.py"

            if not script_path.is_file():
                print(f"Error: {script_path} not found.")
                sys.exit(1)
            
            print("\nLaunching Wikidata extraction...")
            subprocess.run([sys.executable, str(script_path)], check=False)
            break

        elif choice == "4":
            from lib.constants import DEEPL, MODEL_LABEL
            from lib.translations import translate_prompt, translate_rainbow_meter
            print("\nSelect a primary translation model [if you choose DeepL, you'll need to provide another model for the Montenegrin language translation]:")
            primary_model = select_model()
            secondary_model = ""
            if primary_model == DEEPL:
                print("\nSelect a secondary translation model for the Montenegrin language translation:")
                secondary_model = select_model(False)

            translate_rainbow_meter(primary_model, secondary_model)
            break
        
        elif choice == "5":
            from lib.constants import DEEPL, MODEL_LABEL
            from lib.translations import translate_prompt, translate_rainbow_meter
            print("\nSelect a primary translation model [if you choose DeepL, you'll need to provide another model for the Montenegrin language translation]:")
            primary_model = select_model()
            secondary_model = ""
            if primary_model == DEEPL:
                print("\nSelect a secondary translation model for the Montenegrin language translation:")
                secondary_model = select_model(False)
                
            translate_prompt(primary_model, secondary_model)
            break
        
        elif choice == "6":
            from lib.constants import DEEPL, MODEL_LABEL
            from lib.back_translation import back_translation_test
            print("\nSelect a model to perform back translation test:")
            primary_model = select_model()
            
            back_translation_test(primary_model)
            break
        
        elif choice == "7":
            from lib.prompting import Rainbow_Meter
            print("\nSelect a model to prompt the Rainbow Meter questions to:")
            primary_model = select_model()
            
            Rainbow_Meter(primary_model)
            break
        
        elif choice == "8":
            script_path = BASE_DIR / "lib/evaluations.py"

            if not script_path.is_file():
                print(f"Error: {script_path} not found.")
                sys.exit(1)
            print("\nEvaluations in progess...")
            subprocess.run([sys.executable, str(script_path)], check=False)
            break
        
        elif choice == "9":
            script_path = BASE_DIR / "lib/graphs.py"

            if not script_path.is_file():
                print(f"Error: {script_path} not found.")
                sys.exit(1)
            print("\nGraphs generation...")
            subprocess.run([sys.executable, str(script_path)], check=False)
            break
        
        elif choice == "0":
            print("Exiting.")
            break

        elif choice not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            print("Invalid choice. Enter a number beteen 0 and 9.")
