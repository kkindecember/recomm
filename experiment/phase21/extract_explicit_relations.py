"""21.14 first stage: deliberately reads only product catalogues, never user outcomes."""
import json
import time
from pathlib import Path

from analyze_selective_evidence import sha,write
from explicit_relation_core import ExplicitRelations,GENERIC,CUES,ORDINALS

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/explicit_relations_20260925_v1'
RAW=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
PROTOCOL=ROOT/'plan/第二十一阶段/21.14_明示系列与适配关系可用性_20260925.md'


def main():
    t=time.monotonic();OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),Path(__file__).with_name('explicit_relation_core.py'),
        Path(__file__).with_name('analyze_history_evidence.py'),PROTOCOL]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
    (OUT/'protocol.md').write_text(PROTOCOL.read_text())
    for p in paths[:3]:(OUT/p.name).write_text(p.read_text())
    write(OUT/'definitions.json',dict(generic_words=sorted(GENERIC),cues=CUES,ordinals=sorted(ORDINALS)))
    result={}
    for d in ('Toys','Beauty'):
        p=RAW/d/'catalog.json';manifest[str(p.relative_to(ROOT))]=sha(p)
        source=json.loads((RAW/'manifest.json').read_text())['sha256']
        assert manifest[str(p.relative_to(ROOT))]==source[str(p.relative_to(ROOT))]
        engine=ExplicitRelations(json.loads(p.read_text()),d);value=engine.export();write(OUT/(d+'_relations.json'),value)
        result[d]={k:value[k] for k in ('catalog_items','coverage','extraction_audit')}
        print(d,result[d],flush=True)
    write(OUT/'extraction_summary.json',dict(domains=result,seconds=time.monotonic()-t,outcome_files_read=False))
    write(OUT/'manifest.json',manifest)
    write(OUT/'status.json',dict(state='AWAITING_SOURCE_AUDIT',outcome_files_read=False,GRAM_optimizer_steps=0,
        expert_optimizer_steps=0,gpu_inferences=0,test_read=False))


if __name__=='__main__':main()
