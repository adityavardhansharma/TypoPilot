from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "experiments/llama32-1b-lora-001/native_suite_eval_q4/predictions.jsonl"
DEFAULT_REPORT = ROOT / "experiments/llama32-1b-lora-001/native_suite_eval_q4/report.json"
DEFAULT_OUTPUT = ROOT / "reports/llama-q4-900-eval-annotated.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the annotated 900-example Llama evaluation report.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if len(rows) != 900:
        raise ValueError(f"Expected 900 predictions, found {len(rows)}")

    data_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    report_json = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("__PREDICTIONS__", data_json).replace("__REPORT__", report_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} examples)")


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TypoPilot — Llama Q4 900-Example Annotated Evaluation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Zilla+Slab:wght@600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#070b09;--panel:#0c120e;--panel2:#101711;--edge:#18251c;--ink:#c9dccf;--dim:#65796a;--green:#46e08a;--yellow:#e0b346;--red:#e05a46;--cyan:#63d6d1;--pink:#ff72c6;--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;--display:"Zilla Slab",Georgia,serif}
*{box-sizing:border-box;margin:0;padding:0}html{scroll-behavior:smooth}body{background:var(--bg);color:var(--ink);font:13px/1.6 var(--mono)}body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:9;background:repeating-linear-gradient(0deg,rgba(255,255,255,.015) 0 1px,transparent 1px 3px)}
.wrap{max-width:1760px;margin:auto;padding:28px 24px 70px}header{display:flex;align-items:baseline;gap:16px;border-bottom:1px solid var(--edge);padding-bottom:18px;margin-bottom:18px;flex-wrap:wrap}h1{font:700 29px var(--display)}h1 em{color:var(--green);font-style:normal}.tag{font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--bg);background:var(--green);padding:4px 9px;border-radius:3px;font-weight:600}.sub{color:var(--dim);font-size:12px;line-height:1.75;margin-bottom:17px;max-width:1300px}
.legend{display:flex;gap:10px 22px;margin-bottom:22px;color:var(--dim);font-size:12px;flex-wrap:wrap}.mark-e,.mark-g,.mark-y,.mark-x{border-radius:3px;padding:1px 4px;box-decoration-break:clone;-webkit-box-decoration-break:clone}.mark-e{background:rgba(224,90,70,.22);color:#f0937f}.mark-g{background:rgba(70,224,138,.18);color:var(--green)}.mark-y{background:rgba(224,179,70,.2);color:var(--yellow)}.mark-x{background:rgba(255,114,198,.18);color:var(--pink)}
.summary{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:1px;background:var(--edge);border:1px solid var(--edge);margin-bottom:18px}.scard{background:var(--panel);padding:14px 16px;border-top:2px solid var(--accent,var(--green))}.scard b{display:block;color:var(--accent,var(--green));font-size:9px;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:5px}.score{font-size:22px;font-weight:600}.score small{font-size:10px;color:var(--dim);font-weight:400}.meta{margin-top:5px;color:var(--dim);font-size:9px}
.toolbar{position:sticky;top:0;z-index:8;background:rgba(7,11,9,.96);backdrop-filter:blur(10px);border:1px solid var(--edge);padding:12px;margin-bottom:18px;display:grid;grid-template-columns:minmax(240px,1fr) 200px 210px auto;gap:10px}.control{background:var(--panel);border:1px solid #294032;color:var(--ink);font:12px var(--mono);padding:9px 11px;border-radius:3px;outline:none}.control:focus{border-color:var(--green)}.count{display:flex;align-items:center;justify-content:flex-end;color:var(--dim);font-size:11px;white-space:nowrap}.count b{color:var(--green);margin-right:4px}
.case{background:var(--panel);border:1px solid var(--edge);margin-bottom:14px}.case.overcorrected{border-left:2px solid var(--red)}.case.exact{border-left:2px solid var(--green)}.case-head{padding:10px 15px;border-bottom:1px solid var(--edge);display:flex;align-items:center;gap:12px;flex-wrap:wrap}.case-head h2{font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:var(--cyan);font-weight:600}.category{font-size:9px;color:var(--dim);letter-spacing:.8px;text-transform:uppercase}.badges{margin-left:auto;display:flex;align-items:center;gap:7px;flex-wrap:wrap}.badge{border:1px solid currentColor;border-radius:10px;padding:1px 7px;font-size:8px;letter-spacing:.6px;text-transform:uppercase}.pass{color:var(--green)}.warn{color:var(--yellow)}.fail{color:var(--red)}.neutral{color:var(--dim)}.timing{font-size:10px;color:var(--dim)}.timing b{color:var(--green)}
.rows{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:var(--edge)}.text-row{background:var(--panel);padding:13px 15px;min-width:0;white-space:pre-wrap;overflow-wrap:anywhere}.text-row.output{background:var(--panel2)}.text-row b{display:block;color:var(--dim);font-size:8px;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:7px}.text{font-size:12px;line-height:1.85}.empty{padding:70px 20px;text-align:center;color:var(--dim)}.footer{text-align:center;color:var(--dim);font-size:10px;margin-top:25px}.jump{position:fixed;right:20px;bottom:18px;width:36px;height:36px;border:1px solid var(--edge);background:var(--panel);color:var(--green);border-radius:50%;cursor:pointer;z-index:10}
@media(max-width:1100px){.summary{grid-template-columns:repeat(3,1fr)}.rows{grid-template-columns:1fr}.toolbar{grid-template-columns:1fr 1fr}.count{justify-content:flex-start}}@media(max-width:650px){.wrap{padding:18px 10px 55px}.summary,.toolbar{grid-template-columns:1fr}.count{justify-content:flex-start}}
</style>
</head>
<body><div class="wrap" id="top">
<header><h1>Typo<em>Pilot</em> // 900-example evaluation</h1><span class="tag">fine-tuned Llama 3.2 1B · Q4_K_M · CPU</span></header>
<p class="sub">Every test-suite output from the final native CPU run. The center column is the target reference—not necessarily the only acceptable correction. Use the filters to inspect exact matches, remaining errors, meaning-changing overcorrections, and individual categories.</p>
<div class="legend"><span><span class="mark-e">error</span> in input</span><span><span class="mark-g">fixed correctly</span> by the model</span><span><span class="mark-y">still wrong</span>, missing, or wrongly changed</span><span><span class="mark-x">extra context</span> generated beyond the correction</span></div>
<div class="summary" id="summary"></div>
<div class="toolbar">
  <input class="control" id="search" type="search" placeholder="Search ID, input, reference, or output…">
  <select class="control" id="category"><option value="all">All categories</option></select>
  <select class="control" id="status"><option value="all">All outcomes</option><option value="exact">Exact match</option><option value="improved">Improved</option><option value="overcorrected">Overcorrected</option><option value="undercorrected">Undercorrected</option><option value="unchanged">Model left unchanged</option><option value="no-change">No-change reference</option></select>
  <div class="count"><b id="visibleCount">900</b> of 900 shown</div>
</div>
<main id="cases"></main><div class="footer">Generated from the saved 900-example native evaluation · model loading excluded from per-example latency</div>
</div><button class="jump" onclick="scrollTo({top:0,behavior:'smooth'})" title="Back to top">↑</button>
<script>
const DATA=__PREDICTIONS__;
const REPORT=__REPORT__;
const esc=s=>String(s).replace(/[&<>\"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'\"':"&quot;"}[c]));
const words=s=>String(s).trim()?String(s).trim().split(/\s+/):[];
function align(a,b){
 const n=a.length,m=b.length,dp=Array.from({length:n+1},()=>Array(m+1).fill(0));
 for(let i=0;i<=n;i++)dp[i][0]=i;for(let j=0;j<=m;j++)dp[0][j]=j;
 for(let i=1;i<=n;i++)for(let j=1;j<=m;j++)dp[i][j]=Math.min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(a[i-1]===b[j-1]?0:1));
 const ops=[];let i=n,j=m;
 while(i||j){
  if(i&&j&&dp[i][j]===dp[i-1][j-1]+(a[i-1]===b[j-1]?0:1)){ops.push({type:a[i-1]===b[j-1]?"match":"sub",ai:i-1,bj:j-1});i--;j--}
  else if(i&&dp[i][j]===dp[i-1][j]+1){ops.push({type:"del",ai:i-1,bj:null});i--}
  else{ops.push({type:"ins",ai:null,bj:j-1});j--}
 }
 return ops.reverse();
}
function inputMarked(input,reference){
 const iw=words(input),rw=words(reference),out=[];
 for(const op of align(iw,rw))if(op.ai!==null){const t=esc(iw[op.ai]);out.push(op.type==="match"?t:`<span class="mark-e">${t}</span>`)}
 return out.join(" ");
}
function outputMarked(input,reference,generated){
 const split=generated.indexOf("\n\n"),main=split<0?generated:generated.slice(0,split),extra=split<0?"":generated.slice(split+2);
 const iw=words(input),rw=words(reference),ow=words(main),refChanged=Array(rw.length).fill(true);
 for(const op of align(iw,rw))if(op.bj!==null)refChanged[op.bj]=op.type!=="match";
 const pieces=[],missing=[];
 for(const op of align(ow,rw)){
  if(op.ai!==null){const token=esc(ow[op.ai]);if(op.type==="match")pieces.push(refChanged[op.bj]?`<span class="mark-g">${token}</span>`:token);else pieces.push(`<span class="mark-y">${token}</span>`)}
  else if(op.bj!==null)missing.push(rw[op.bj]);
 }
 if(missing.length)pieces.push(`<span class="mark-y">⟨missing: ${esc(missing.join(" "))}⟩</span>`);
 if(extra)pieces.push(`\n\n<span class="mark-x">${esc(extra)}</span>`);
 return pieces.join(" ").replace(/ \n\n/g,"\n\n");
}
function outcome(d){if(d.exact_match)return["pass","exact"];if(d.overcorrected)return["fail","overcorrected"];if(d.improvement>0)return["warn","improved, not exact"];if(d.undercorrected)return["warn","undercorrected"];return["neutral","unchanged distance"]}
function card(d){const [cls,label]=outcome(d);return `<article class="case ${d.exact_match?"exact":""} ${d.overcorrected?"overcorrected":""}" data-id="${esc(d.id)}"><div class="case-head"><h2>${esc(d.id)}</h2><span class="category">${esc(d.category.replaceAll("_"," "))}</span><div class="badges"><span class="badge ${cls}">${label}</span><span class="timing"><b>${Math.round(d.latency_ms)}</b>ms · CER ${(d.cer*100).toFixed(2)}%</span></div></div><div class="rows"><div class="text-row"><b>input</b><div class="text">${inputMarked(d.input,d.reference)}</div></div><div class="text-row"><b>reference</b><div class="text">${esc(d.reference)}</div></div><div class="text-row output"><b>fine-tuned Llama Q4 output</b><div class="text">${outputMarked(d.input,d.reference,d.hypothesis)}</div></div></div></article>`}
function summary(){const o=REPORT.overall,items=[["exact match",`${(o.exact_match*100).toFixed(2)}%`,`${o.exact_match_count} / ${o.n}`,"var(--green)"],["improved",`${(o.improved_count/o.n*100).toFixed(2)}%`,`${o.improved_count} examples`,"var(--cyan)"],["output CER",`${(o.output_cer*100).toFixed(2)}%`,`input ${(o.input_cer*100).toFixed(2)}%`,"var(--green)"],["overcorrected",`${(o.overcorrected*100).toFixed(2)}%`,`${o.overcorrected_count} examples`,"var(--red)"],["no-change kept",`${(o.no_change_preservation_rate*100).toFixed(2)}%`,`${o.no_change_examples-o.no_change_changed_count} / ${o.no_change_examples}`,"var(--yellow)"],["CPU latency",`${Math.round(o.latency_ms)}ms`,`p50 ${Math.round(o.latency_p50_ms)}ms · p95 ${Math.round(o.latency_p95_ms)}ms`,"var(--cyan)"]];document.getElementById("summary").innerHTML=items.map(x=>`<div class="scard" style="--accent:${x[3]}"><b>${x[0]}</b><div class="score">${x[1]}</div><div class="meta">${x[2]}</div></div>`).join("")}
const search=document.getElementById("search"),category=document.getElementById("category"),status=document.getElementById("status"),cases=document.getElementById("cases"),visibleCount=document.getElementById("visibleCount");
function matches(d){const q=search.value.trim().toLowerCase(),c=category.value,s=status.value;if(c!=="all"&&d.category!==c)return false;if(q&&!`${d.id}\n${d.input}\n${d.reference}\n${d.hypothesis}`.toLowerCase().includes(q))return false;if(s==="exact"&&!d.exact_match)return false;if(s==="improved"&&!(d.improvement>0))return false;if(s==="overcorrected"&&!d.overcorrected)return false;if(s==="undercorrected"&&!d.undercorrected)return false;if(s==="unchanged"&&d.input!==d.hypothesis)return false;if(s==="no-change"&&d.input!==d.reference)return false;return true}
function render(){const rows=DATA.filter(matches);visibleCount.textContent=rows.length;cases.innerHTML=rows.length?rows.map(card).join(""):`<div class="empty">No examples match these filters.</div>`}
for(const c of [...new Set(DATA.map(d=>d.category))])category.insertAdjacentHTML("beforeend",`<option value="${esc(c)}">${esc(c.replaceAll("_"," "))}</option>`);
summary();render();let timer;search.addEventListener("input",()=>{clearTimeout(timer);timer=setTimeout(render,100)});category.addEventListener("change",render);status.addEventListener("change",render);
</script></body></html>'''


if __name__ == "__main__":
    main()
