#!/bin/bash
# Development setup script for multidocqa project

set -e  # Exit on any error

echo "🚀 Setting up development environment for multidocqa..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install/update development dependencies
echo "📋 Installing development dependencies..."
uv pip install -e ".[dev]"

# Install pre-commit hooks
echo "🎣 Installing pre-commit hooks..."
pre-commit install

# Format existing code
echo "🎨 Formatting existing code with black..."
black multidocqa/

# Sort imports
echo "📂 Sorting imports with isort..."
isort multidocqa/

# Run initial checks
echo "🔍 Running initial code quality checks..."
echo "  - Black formatting check..."
if black --check multidocqa/; then
    echo "    ✅ Code formatting looks good"
else
    echo "    ℹ️  Some files were reformatted"
fi

echo "  - Flake8 linting..."
if flake8 multidocqa/; then
    echo "    ✅ No linting issues found"
else
    echo "    ⚠️  Some linting issues found (see above)"
fi

echo "  - Import sorting check..."
if isort --check-only multidocqa/; then
    echo "    ✅ Import sorting looks good"
else
    echo "    ℹ️  Some imports were sorted"
fi

echo "  - Running tests..."
if pytest multidocqa/tests/ -v; then
    echo "    ✅ All tests passed"
else
    echo "    ⚠️  Some tests failed (see above)"
fi

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "📝 What happens now:"
echo "   • When you commit code, pre-commit hooks will automatically:"
echo "     - Format code with black"
echo "     - Sort imports with isort"
echo "     - Run flake8 linting"
echo "     - Run mypy type checking"
echo "     - Check for common issues"
echo ""
echo "🛠️  Manual commands you can run:"
echo "   • source .venv/bin/activate    - Activate virtual environment"
echo "   • black multidocqa/           - Format code"
echo "   • flake8 multidocqa/          - Lint code"
echo "   • isort multidocqa/           - Sort imports"
echo "   • pytest multidocqa/tests/    - Run tests"
echo "   • pre-commit run --all-files  - Run all hooks manually"
echo ""
echo "🎯 Ready to code!"
