#!/bin/bash
#SBATCH --job-name=rainbow_meter
#SBATCH --partition=gpu_a100
#SBATCH --time=00-00:10:00
#SBATCH --gres=gpu:ha100:1
#SBATCH --mem=20G
#SBATCH --output=./slurmout/rainbow_meter.out
#SBATCH --error=./slurmout/rainbow_meter.err

set -e

echo "🐍 Creating Python virtual environment..."
python3 -m venv .venv > /dev/null
source .venv/bin/activate > /dev/null

echo "🔧 Installing base libraries..."
pip install torch pandas tqdm transformers deep-translator SPARQLWrapper unidecode surprisal transformers python-dotenv > /dev/null

echo "🤖 Installing model APIs..."
pip install openai > /dev/null
pip install vllm > /dev/null

for pkg in openai google-generativeai; do
    echo "   → Installing $pkg..."
    pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
done

echo "🔧 Installing base libraries..."
for pkg in torch pandas tqdm unidecode surprisal transformers python-dotenv; do
    echo "   → Installing $pkg..."
    pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
done

echo "📊 Installing graph & ML libraries..."
for pkg in tf-keras seaborn scikit-learn scipy matplotlib wordcloud python-ternary; do
    echo "   → Installing $pkg..."
    pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
done

echo "✅ Environment setup complete!"