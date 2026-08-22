"""Run Cycle B6 evidence-triggered deep gating experiment."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from gated_residuals.artifacts import RunMetadata, save_manifest, save_records_csv, save_records_parquet
from gated_residuals.common.config import read_yaml
from gated_residuals.experiments import confidence, current_commit, load_seed, make_loader, resolve_device, train_seed
from gated_residuals.standard_metrics import target_log_probability


def parse_args():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/b6_gated_deep.yaml"); parser.add_argument("--output", default=None); parser.add_argument("--fresh", action="store_true"); parser.add_argument("--postprocess-only", action="store_true"); return parser.parse_args()


def finite(value):
    result = float(value)
    if not math.isfinite(result): raise RuntimeError("non-finite B6 metric")
    return result


@torch.inference_mode()
def validation_gate_means(model, dataset, config, seed, device):
    sums = {"block": [0.0] * model.num_layers, "sa": [0.0] * model.num_layers, "ff": [0.0] * model.num_layers}; count = 0
    for batch in make_loader(dataset, config, shuffle=False, seed=seed):
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), capture=True); count += out.logits.size(0)
        for layer in range(model.num_layers):
            sums["block"][layer] += float(out.gates[layer].sum())
            sums["sa"][layer] += float(out.attention_gates[layer].sum())
            sums["ff"][layer] += float(out.ff_gates[layer].sum())
    return {name: [value / count for value in values] for name, values in sums.items()}


def overrides(values, batch_size, device, dtype):
    return [torch.full((batch_size, 1), float(value), device=device, dtype=dtype) for value in values]


@torch.inference_mode()
def analyze(model, splits, datasets, config, variant, seed):
    device_name = resolve_device(config["training"].get("device", "auto")); device = torch.device(device_name); model.to(device).eval()
    means = validation_gate_means(model, datasets["val"], config, seed, device)
    analysis_config = json.loads(json.dumps(config)); analysis_config["training"]["batch_size"] = min(32, int(config["training"]["batch_size"]))
    quality, causal = [], []; max_parity = 0.0
    for batch in make_loader(datasets["test"], analysis_config, shuffle=False, seed=seed):
        ids=batch["input_ids"].to(device); mask=batch["attention_mask"].to(device); targets=batch["target"].to(device)
        plain=model(ids,mask); native=model(ids,mask,capture=True); parity=float((plain.logits-native.logits).abs().max()); max_parity=max(max_parity,parity)
        if not torch.equal(plain.logits,native.logits): raise RuntimeError(f"B6 native/capture parity failed: {parity}")
        modes={"native":native,"open":model(ids,mask,gate_mode="open"),"closed":model(ids,mask,gate_mode="closed")}
        if variant == "gated":
            modes["mean"]=model(ids,mask,gate_overrides=overrides(means["block"],ids.size(0),device,native.logits.dtype))
            modes["shuffled"]=model(ids,mask,gate_overrides=[gate.roll(1,0) for gate in native.gates])
        else:
            modes["mean"]=model(ids,mask,attention_gate_overrides=overrides(means["sa"],ids.size(0),device,native.logits.dtype),ff_gate_overrides=overrides(means["ff"],ids.size(0),device,native.logits.dtype))
            modes["shuffled"]=model(ids,mask,attention_gate_overrides=[gate.roll(1,0) for gate in native.attention_gates],ff_gate_overrides=[gate.roll(1,0) for gate in native.ff_gates])
        mode_stats={name:(target_log_probability(out.logits,targets).cpu(),out.logits.argmax(-1).cpu()) for name,out in modes.items()}
        full_lp=mode_stats["native"][0]
        intervention=[]
        for layer in range(model.num_layers):
            sa=model(ids,mask,skip_attention_layers={layer}); ff=model(ids,mask,skip_ff_layers={layer}); block=model(ids,mask,skip_layers={layer})
            intervention.append((target_log_probability(sa.logits,targets).cpu(),target_log_probability(ff.logits,targets).cpu(),target_log_probability(block.logits,targets).cpu()))
        for row,index in enumerate(batch["example_index"].tolist()):
            example=splits["test"][index]; length=int(mask[row].sum()); token=length-1
            for mode,(lp,prediction) in mode_stats.items():
                quality.append({"variant":variant,"seed":seed,"task_family":example.intent,"example_id":example.example_id,"mode":mode,"correct":bool(prediction[row]==targets[row].cpu()),"target_logprob":finite(lp[row])})
            for layer in range(model.num_layers):
                sa_lp,ff_lp,block_lp=intervention[layer]
                candidate=native.candidates[layer][row,token].float(); effective=native.effective_updates[layer][row,token].float()
                causal.append({"cycle":"B","experiment":"B6","revision":int(config["revision"]),"variant":variant,"seed":seed,"task_family":example.intent,"example_id":example.example_id,"layer":layer,"full_correct":bool(mode_stats["native"][1][row]==targets[row].cpu()),"full_target_logprob":finite(full_lp[row]),"utility_block":finite(full_lp[row]-block_lp[row]),"utility_sa":finite(full_lp[row]-sa_lp[row]),"utility_ff":finite(full_lp[row]-ff_lp[row]),"gate_block":finite(native.gates[layer][row]),"gate_sa":finite(native.attention_gates[layer][row]),"gate_ff":finite(native.ff_gates[layer][row]),"candidate_update_norm":finite(torch.linalg.vector_norm(candidate)),"effective_update_norm":finite(torch.linalg.vector_norm(effective)),"effective_candidate_ratio":finite(torch.linalg.vector_norm(effective)/torch.linalg.vector_norm(candidate).clamp_min(1e-8))})
    return quality,causal,max_parity,means,device_name


def summarize(quality,causal,config):
    q=pd.DataFrame(quality); c=pd.DataFrame(causal); baseline=pd.read_parquet("artifacts/cycle_b/b5_task_ecology/quality_records.parquet")
    base_seed=baseline.groupby(["seed","task_family"],as_index=False).correct.mean().rename(columns={"correct":"baseline_accuracy"})
    seed_quality=q.groupby(["variant","mode","seed","task_family"],as_index=False).correct.mean().merge(base_seed,on=["seed","task_family"],how="left")
    seed_quality["accuracy_minus_baseline"]=seed_quality.correct-seed_quality.baseline_accuracy
    quality_rows=[]
    for (variant,mode,task),group in seed_quality.groupby(["variant","mode","task_family"]):
        row={"variant":variant,"mode":mode,"task_family":task,"seed_n":len(group)}
        for metric in ("correct","accuracy_minus_baseline"):
            mean,low,high=confidence(group[metric].tolist(),seed=len(variant)+len(mode)+len(task)); row[f"{metric}_mean"]=mean;row[f"{metric}_ci95_low"]=low;row[f"{metric}_ci95_high"]=high
        quality_rows.append(row)
    gate_rows=[]
    for (variant,task,layer,seed),group in c.groupby(["variant","task_family","layer","seed"]):
        row={"variant":variant,"task_family":task,"layer":int(layer),"seed":int(seed),"n":len(group)}
        for component in ("block","sa","ff"):
            gate=group[f"gate_{component}"]; utility=group[f"utility_{component}"]
            row[f"gate_{component}_mean"]=float(gate.mean()); row[f"gate_{component}_below_0_5"]=float((gate<.5).mean()); row[f"gate_{component}_above_0_9"]=float((gate>.9).mean()); row[f"utility_{component}_mean"]=float(utility.mean()); row[f"gate_utility_{component}_spearman"]=finite(gate.corr(utility,method="spearman")) if gate.nunique()>1 and utility.nunique()>1 else None
        row["gate_candidate_norm_spearman"]=finite(group.gate_block.corr(group.candidate_update_norm,method="spearman")) if group.gate_block.nunique()>1 else None
        gate_rows.append(row)
    gates=pd.DataFrame(gate_rows); summary_rows=[]
    for (variant,task,layer),group in gates.groupby(["variant","task_family","layer"]):
        row={"variant":variant,"task_family":task,"layer":int(layer),"seed_n":len(group)}
        for metric in [x for x in gates.columns if x not in {"variant","task_family","layer","seed","n"}]:
            values=group[metric].dropna().tolist()
            if values:
                mean,low,high=confidence(values,seed=int(layer)+len(metric)+len(task));row[f"{metric}_mean"]=mean;row[f"{metric}_ci95_low"]=low;row[f"{metric}_ci95_high"]=high
            else: row[f"{metric}_mean"]=None;row[f"{metric}_ci95_low"]=None;row[f"{metric}_ci95_high"]=None
        summary_rows.append(row)
    return seed_quality.to_dict("records"),quality_rows,gate_rows,summary_rows


def plot(quality_rows,gate_rows,output):
    q=pd.DataFrame(quality_rows); native=q[q["mode"].eq("native")].groupby("variant").correct_mean.mean(); g=pd.DataFrame(gate_rows)
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.4)); axes[0].bar(native.index,native.values,color=["#4472c4","#ed7d31"]);axes[0].axhline(.988,color="black",linestyle="--",linewidth=1,label="B5 baseline");axes[0].set(ylabel="Mean test accuracy",ylim=(0,1.02));axes[0].legend(frameon=False)
    for variant,group in g.groupby("variant"):
        profile=group.groupby("layer").gate_block_mean.mean();axes[1].plot(profile.index,profile.values,marker="o",markersize=2,label=variant)
    axes[1].set(xlabel="Layer",ylabel="Mean applied gate",ylim=(0,1.02));axes[1].legend(frameon=False);fig.tight_layout();(output/"figures").mkdir(parents=True,exist_ok=True);fig.savefig(output/"figures"/"b6_gating.pdf",bbox_inches="tight");fig.savefig(output/"figures"/"b6_gating.png",dpi=180,bbox_inches="tight");plt.close(fig)


def main():
    args=parse_args();config=read_yaml(args.config);output=Path(args.output or config["output"]["directory"])
    if output.exists() and args.fresh: shutil.rmtree(output)
    output.mkdir(parents=True,exist_ok=True)
    b4=json.loads(Path(config["evidence_sources"][0]).read_text());b5=json.loads(Path(config["evidence_sources"][1]).read_text())
    if not (b4["b6_gate_open_from_b4"] and b5["b6_gate_open_from_b5"]): raise RuntimeError("B6 evidence gate is closed")
    all_quality=[];all_causal=[];registry=[];started=time.perf_counter()
    if args.postprocess_only:
        all_quality=pd.read_parquet(output/"quality_records.parquet").to_dict("records")
        all_causal=pd.read_parquet(output/"causal_gate_records.parquet").to_dict("records")
        registry=pd.read_csv(output/"run_registry.csv").to_dict("records")
    for variant in ([] if args.postprocess_only else config["variants"]):
        for seed in config["training"]["seeds"]:
            run_dir=output/"runs"/variant/f"seed-{seed}";checkpoint=run_dir/"model.pt"
            if checkpoint.exists(): model,splits,vocabulary,datasets=load_seed(config,checkpoint);history=json.loads((run_dir/"history.json").read_text());training={"training_seconds":0.0,"best_val_loss":min(x["val_loss"] for x in history)}
            else: model,splits,vocabulary,datasets,training=train_seed(config,variant,int(seed),run_dir)
            quality,causal,parity,means,device=analyze(model,splits,datasets,config,variant,int(seed));all_quality.extend(quality);all_causal.extend(causal)
            registry.append({"cycle":"B","experiment":"B6","revision":int(config["revision"]),"run_id":f"{variant}-seed-{seed}","model_family":"tiny_custom_transformer","model_variant":variant,"depth":model.num_layers,"width":model.width,"task_family":"+".join(config["data"]["intents"]),"seed":int(seed),"checkpoint":str(checkpoint).replace("\\","/"),"parameters":sum(p.numel() for p in model.parameters()),"test_examples":len(splits["test"]),"native_capture_max_logit_error":parity,"best_val_loss":training["best_val_loss"],"training_seconds":training["training_seconds"],"validation_gate_means":json.dumps(means),"device":device});del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
    seed_quality,quality_rows,gate_rows,summary_rows=summarize(all_quality,all_causal,config)
    save_records_parquet(output/"quality_records.parquet",all_quality);save_records_parquet(output/"causal_gate_records.parquet",all_causal);save_records_csv(output/"run_registry.csv",registry);save_records_csv(output/"seed_quality.csv",seed_quality);save_records_csv(output/"quality_summary.csv",quality_rows);save_records_csv(output/"gate_seed_summary.csv",gate_rows);save_records_csv(output/"gate_summary.csv",summary_rows);plot(quality_rows,gate_rows,output)
    q=pd.DataFrame(quality_rows);g=pd.DataFrame(gate_rows);native=q[q["mode"].eq("native")];open_q=q[q["mode"].eq("open")]
    result={"cycle":"B","experiment":"B6","revision":int(config["revision"]),"source_commit":current_commit(),"runs":len(registry),"quality_rows":len(all_quality),"causal_rows":len(all_causal),"native_capture_max_logit_error":max(x["native_capture_max_logit_error"] for x in registry),"native_accuracy_by_variant":native.groupby("variant").correct_mean.mean().to_dict(),"native_minus_baseline_by_variant":native.groupby("variant").accuracy_minus_baseline_mean.mean().to_dict(),"open_minus_native_by_variant":{variant:float(open_q[open_q.variant.eq(variant)].correct_mean.mean()-native[native.variant.eq(variant)].correct_mean.mean()) for variant in config["variants"]},"mean_gate_by_variant":g.groupby("variant").gate_block_mean.mean().to_dict(),"mean_gate_utility_spearman_by_variant":g.groupby("variant").gate_utility_block_spearman.mean().to_dict(),"mean_gate_candidate_norm_spearman_by_variant":g.groupby("variant").gate_candidate_norm_spearman.mean().to_dict(),"minimum_layer8":g[(g.task_family.eq("minimum"))&(g.layer.eq(8))].groupby("variant")[["gate_block_mean","gate_ff_mean","utility_block_mean","utility_ff_mean"]].mean().to_dict("index"),"elapsed_seconds":None if args.postprocess_only else time.perf_counter()-started,"postprocess_elapsed_seconds":time.perf_counter()-started if args.postprocess_only else None,"interpretation":"Gates are interpreted as computational selection only when they align with independent causal utility or improve quality under interventions; soft gates do not imply FLOP savings."}
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");(output/"failure_null_notes.md").write_text("# B6 failure and null-result notes\n\n- B6 was opened by B4/B5 causal evidence, not by retrospective gate behavior.\n- Soft gates evaluate every candidate update and save no realized FLOPs.\n- Shuffled controls operate within deterministic test batches; mean controls use validation-only gate means.\n- Gate correlation with utility is descriptive unless supported by interventions and seed replication.\n",encoding="utf-8")
    metadata=RunMetadata.collect(run_id="cycle-b-b6",model="tiny_residual_decoder",model_variant="gated+sa_ff_gated",dataset="synthetic_counterfactual_v2_selection",seed=0,context_length=int(config["data"]["max_length"]),batch_size=int(config["training"]["batch_size"]),dtype="float32",device=registry[0]["device"]);save_manifest(output/"manifest.json",metadata,config);elapsed_label="null" if result["elapsed_seconds"] is None else f"{result['elapsed_seconds']:.3f}";(output/"run.log").write_text(f"B6 complete runs={len(registry)} elapsed_seconds={elapsed_label} postprocess_elapsed_seconds={result['postprocess_elapsed_seconds']:.3f}\n",encoding="utf-8");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
