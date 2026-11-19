"""Derived plot of both predeclared diagnostic-input origins; no new experiments."""
import argparse
import numpy as np

from project_paths import local_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_gradient_deletion.contracts import atomic_csv
from scripts.adaptation_gradient_pilot.methods import require
from scripts.adaptation_stage7.runtime import seal,verify


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',default='results/experiments/adaptation_stage7_suite_v1')
    args=parser.parse_args(argv);root=local_path(args.output_root,must_exist=True)
    require(verify(root)['stage7_scoped_complete'],'Verified parent scope required')
    destination=root/'origin_controls_v1'
    if (destination/'COMPLETE.json').exists():verify(destination);return 0
    destination.mkdir(exist_ok=True)
    manifest=dict(parent_complete_sha256=pilot.sha256(root/'COMPLETE.json'),aggregates_sha256=pilot.sha256(root/'SEED_AGGREGATES.json'),
        source_sha256=pilot.sha256(__file__),role='presentation of both predeclared origins; original evidence unchanged')
    if (destination/'MANIFEST.json').exists():pilot.validate_manifest(pilot.read_json(destination/'MANIFEST.json'),manifest)
    else:pilot.write_json(destination/'MANIFEST.json',manifest)
    data=pilot.read_json(root/'SEED_AGGREGATES.json');spec=pilot.read_json(root/'MANIFEST.json')['specification']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    rows=[]
    for ax,ds in zip(axes,spec['datasets']):
        for role,offset,marker in [('parent',-.07,'o'),('variant',.07,'s')]:
            groups=[next(r for r in data if r['dataset']==ds and r['pair']==pair['id'] and r['origin_role']==role) for pair in spec['pairs']]
            ax.errorbar(np.arange(4)+offset,[g['optimized_minus_random']['mean'] for g in groups],
                yerr=[g['optimized_minus_random']['sample_sd'] for g in groups],fmt=marker,capsize=4,label=role+'-optimized input')
            for g in groups:
                rows.append(dict(dataset='SHD*' if ds=='SHD' else ds,pair=g['pair'],origin_role=role,
                    mean_ce_contrast=g['optimized_minus_random']['mean'],sample_sd_training_seeds=g['optimized_minus_random']['sample_sd'],n_training_seeds=3))
        ax.axhline(0,color='black',linewidth=.6);ax.set_xticks(np.arange(4),[p['id'] for p in spec['pairs']],rotation=30)
        ax.set(title='SHD*' if ds=='SHD' else ds,ylabel='Optimized − random benefit change (CE)');ax.legend(fontsize=8)
    fig.suptitle('Diagnostic-input origin changes the contrast: mean ± sample SD across three training seeds\nFour fixed records per seed; conditional model-selected probes, not a natural robustness ranking')
    figure=destination/'both_optimization_origins.png';fig.savefig(figure,dpi=160);plt.close(fig)
    table=destination/'ORIGIN_ROLE_SUMMARY.csv';atomic_csv(table,rows)
    text=['# Both predeclared optimization origins','',
        'Mean ± sample SD across three matching training seeds after averaging four fixed diagnostic records within seed.',
        'CE contrast = change in parent-minus-variant supervised loss under optimized input minus its matched-random change.',
        'Each origin is selected to increase that model’s own CE; the comparison is conditional on this selection.',
        'Parent-origin positive contrasts and variant-origin negative contrasts must be interpreted together. No natural-accuracy or causal superiority follows.','',
        '| Dataset | Pair | Parent-origin CE contrast | Variant-origin CE contrast |','|---|---|---:|---:|']
    for ds in ['SHD*','SSC','ECG']:
        for pair in spec['pairs']:
            groups=[next(r for r in rows if r['dataset']==ds and r['pair']==pair['id'] and r['origin_role']==role) for role in ['parent','variant']]
            text.append(f"| {ds} | {pair['id']} | {groups[0]['mean_ce_contrast']:+.6g} ± {groups[0]['sample_sd_training_seeds']:.6g} | {groups[1]['mean_ce_contrast']:+.6g} ± {groups[1]['sample_sd_training_seeds']:.6g} |")
    summary=destination/'SUMMARY.md';summary.write_text('\n'.join(text)+'\n',encoding='utf-8')
    seal(destination,manifest,[destination/'MANIFEST.json',figure,table,summary],presentation_only=True)
    print('Both-origin presentation verified outputs:',len(verify(destination)['outputs']));return 0


if __name__=='__main__':raise SystemExit(main())
