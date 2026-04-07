#!/usr/bin/env python3
"""
Fast AutoML CLI - Optimized for Large Datasets
Auto-samples to 100K rows and skips slow models
"""
import sys

# Add flags for fast mode
if '--sample' not in sys.argv:
    sys.argv.extend(['--sample', '100000'])
if '--quick' not in sys.argv:
    sys.argv.append('--quick')

# Import and run main CLI
from automl_cli import main
main()
