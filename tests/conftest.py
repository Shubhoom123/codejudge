import sys
from pathlib import Path

# Make the project root importable (config, evaluator, sandbox, trainer.*)
# without requiring the package to be installed.
sys.path.insert(0, str(Path(__file__).parent.parent))
