#!/bin/bash

# Exit immediately on error
set -e

echo "🐍 Creating Python virtual environment..."
python3 -m venv .venv > /dev/null
source .venv/bin/activate > /dev/null

echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null

echo "🔧 Installing base libraries..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pandas tqdm python-dotenv > /dev/null

echo "🤖 Installing model APIs..."
pip install openai > /dev/null

for pkg in openai google-generativeai; do
    echo "   → Installing $pkg..."
    pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
done

echo "📊 Installing graph & ML libraries..."
for pkg in tf-keras seaborn scikit-learn scipy matplotlib wordcloud python-ternary; do
    echo "   → Installing $pkg..."
    pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
done

echo "✅ Environment setup complete!"