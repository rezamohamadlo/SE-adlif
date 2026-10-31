import hashlib, json, os, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
plan=json.loads((OUT/'FROZEN_TEST_PLAN.json').read_text())
source=(ROOT/'test_checkpoint.py').read_text()
needle="    directory = checkpoint_path.parent.parent"
assert source.count(needle)==1
for r in plan['checkpoints']:
    checkpoint=ROOT/r['checkpoint']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['sha256']
    target=OUT/f"seed_{r['seed']}"
    target.mkdir(exist_ok=True)
    if (target/'TEST_RESULTS.json').exists():
        continue
    print(f"Evaluating seed {r['seed']}",flush=True)
    code=source.replace(needle,"    directory = Path("+repr(str(target))+")")
    sys.argv=['test_checkpoint.py','--checkpoint',str(checkpoint)]
    exec(compile(code,str(ROOT/'test_checkpoint.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'test_checkpoint.py')})
rows=[]
for r in plan['checkpoints']:
    result=json.loads((OUT/f"seed_{r['seed']}"/'TEST_RESULTS.json').read_text())
    rows.append(dict(seed=r['seed'],test_accuracy_percent=100*result['metrics']['test_acc']))
values=[r['test_accuracy_percent'] for r in rows]
summary=dict(model=plan['model'],dataset=plan['dataset'],n=len(rows),seeds=rows,mean_percent=statistics.mean(values),sample_sd_percentage_points=statistics.stdev(values))
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
