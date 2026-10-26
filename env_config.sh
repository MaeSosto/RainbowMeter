#!/bin/bash

# Exit immediately on error
set -e

echo "🐍 Creating Python virtual environment..."
python3 -m venv .venv > /dev/null
source .venv/bin/activate > /dev/null

echo "🔧 Installing base libraries..."
pip install torch torchvision tqdm  data scipy SPARQLWrapper python-dotenv data accelerate > /dev/null

echo "🤖 Installing model APIs..."
pip install transformers deep-translator openai deepl anthropic google-genai google-generativeai > /dev/null

echo "📊 Installing graph & ML libraries..."
for pkg in lingua-language-detector seaborn pandas scikit-learn scipy matplotlib python-ternary; do
    echo "   → Installing $pkg..."
    pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
done

echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment manually, run:"
echo "source .venv/bin/activate"
echo ""
echo "To run your application, use:"
echo ".venv/bin/python run.py"