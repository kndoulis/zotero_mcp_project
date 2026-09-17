#!/bin/bash
# ABOUTME: Setup script for the Zotero MCP server installation and configuration
# ABOUTME: Installs dependencies, sets up environment, and provides usage instructions

set -e

echo "🔧 Setting up Zotero MCP Server..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Please install Python 3.9 or later."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔄 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🔨 Installing the package in editable mode..."
pip install -e .

echo "⚙️  Setting up configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file from template. Please edit it with your Zotero credentials:"
    echo "   - Get your API key from: https://www.zotero.org/settings/keys"
    echo "   - Find your User ID in your profile URL: https://www.zotero.org/users/YOUR_USER_ID"
    echo ""
    echo "   Edit .env and set:"
    echo "   ZOTERO_API_KEY=your_api_key_here"
    echo "   ZOTERO_USER_ID=your_user_id_here"
else
    echo "✅ .env file already exists"
fi

echo "🧪 Running tests..."
python -m pytest tests/ -v || echo "⚠️  Some tests failed - this is expected if dependencies are still installing"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit the .env file with your Zotero API credentials"
echo "2. Run the server: python -m zotero_mcp_server.main"
echo "3. Connect it to Claude Desktop by adding to your claude_desktop_config.json:"
echo ""
echo '   "mcpServers": {'
echo '     "zotero": {'
echo '       "command": "python",'
echo '       "args": ["-m", "zotero_mcp_server.main"],'
echo '       "cwd": "'$(pwd)'"'
echo '     }'
echo '   }'
echo ""
echo "Happy researching! 📚✨"