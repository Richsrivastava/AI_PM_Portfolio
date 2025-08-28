import yaml, time, csv
from pathlib import Path

def mock_model(prompt): 
    return "Helpful response including analysis, writing, delay, appreciate, and resolve."

def run(path):
    tasks = yaml.safe_load(open(path))['tasks']; rows=[]
    for t in tasks:
        t0=time.time(); out=mock_model(t.get('prompt','') + " " + t.get('input',''))
        dt=int((time.time()-t0)*1000)
        rows.append({"id":t["id"],"latency_ms":dt,"output":out})
    p=Path(path).with_name("eval_results.csv")
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("Saved", p)

if __name__=="__main__": run(Path(__file__).with_name("evalset.yaml"))
