"""MT-TC labels and explicit training/evaluation objective distinction."""
from scripts.adaptation_analysis.analysis_reporting import write_reports as original_reports


def write_reports(root,paired,metadata):
    paths=original_reports(root,paired,metadata)
    for path in paths:
        if path.suffix=='.md':
            text=path.read_text(encoding='utf-8').replace('favors AMT','favors MT-TC')
            text+='\nTraining objectives intentionally differ: MT-TC was trained with an auxiliary JS penalty. Reported prediction losses and benefits use supervised cross-entropy only, excluding that penalty. Shared baseline settings do not make these training objectives identical.\n'
            path.write_text(text,encoding='utf-8')
    return paths
