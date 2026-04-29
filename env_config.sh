#!/bin/bash

# Exit immediately on error
set -e

echo "🐍 Creating Python virtual environment..."
python3 -m venv .venv > /dev/null
source .venv/bin/activate > /dev/null

# echo "📦 Installing Ollama and starting the service..."
# pip install ollama > /dev/null
# Uncomment the models you want to pull
# ollama pull llama3
# ollama pull llama3.3
# ollama pull llama3:70b
# ollama pull gemma3
# ollama pull gemma3:27b
# ollama pull deepseek-r1
# ollama pull deepseek-r1:70b
# ollama serve > /dev/null &

echo "🔧 Installing base libraries..."
pip install torch tqdm pandas data tqdm scipy SPARQLWrapper unidecode surprisal python-dotenv data accelerate > /dev/null

echo "🤖 Installing model APIs..."
pip install transformers deep-translator openai deepl anthropic google-genai google-generativeai > /dev/null

echo "📊 Installing graph & ML libraries..."
for pkg in tf-keras seaborn scikit-learn scipy matplotlib wordcloud python-ternary; do
    echo "   → Installing $pkg..."
    pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
done

#echo "📦 Upgrading pip..."
#pip install --upgrade pip > /dev/null

echo "✅ Environment setup complete!"