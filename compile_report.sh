#!/bin/bash
# Script per compilare il report LaTeX in PDF

echo "Compiling LaTeX report to PDF..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found. Please install TeX Live or MiKTeX."
    echo ""
    echo "On Ubuntu/Debian:"
    echo "  sudo apt-get install texlive-full"
    echo ""
    echo "On macOS (with Homebrew):"
    echo "  brew install --cask mactex"
    exit 1
fi

# Compile twice for references
echo "First pass..."
pdflatex -interaction=nonstopmode report.tex > /dev/null

echo "Second pass..."
pdflatex -interaction=nonstopmode report.tex > /dev/null

# Clean up auxiliary files
echo "Cleaning up..."
rm -f report.aux report.log report.out report.toc report.lof report.lot

if [ -f "report.pdf" ]; then
    echo "✓ Success! Generated: report.pdf"
    
    # Show file size
    size=$(ls -lh report.pdf | awk '{print $5}')
    echo "  File size: $size"
else
    echo "✗ Error: Failed to generate PDF"
    echo "  Check report.tex for LaTeX errors"
    exit 1
fi
