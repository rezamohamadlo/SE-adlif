"""LT-DTH report labels using the existing paired reductions."""
from scripts.adaptation_analysis.analysis_reporting import write_reports as original_reports


def write_reports(root, paired, metadata):
    paths = original_reports(root, paired, metadata)
    for path in paths:
        if path.suffix == '.md':
            path.write_text(path.read_text(encoding='utf-8').replace('favors AMT', 'favors LT-DTH'),encoding='utf-8')
    return paths
