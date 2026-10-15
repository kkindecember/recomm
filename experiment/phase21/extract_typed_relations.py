"""Source-only extraction and audit materials, before any outcome evaluation."""
import json
import time
from pathlib import Path
from analyze_selective_evidence import sha,write
from typed_relation_core import TypedRelations,BOUNDARIES,DIMENSIONS,SEASONS,NUMBER_WORDS

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/typed_relations_20260925_v1'
PREVIOUS=ROOT/'artifacts/phase21/entity_relations_20260925_v1'
RAW=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
PROTOCOL=ROOT/'plan/第二十一阶段/21.16_实体角色与版本分层固定检查_20260925.md'


def main():
    started=time.monotonic();OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),Path(__file__).with_name('typed_relation_core.py'),Path(__file__).with_name('entity_relation_core.py'),
           Path(__file__).with_name('analyze_history_evidence.py'),PROTOCOL]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for p in paths:(OUT/p.name).write_text(p.read_text())
    write(OUT/'definitions.json',dict(boundaries=sorted(BOUNDARIES),dimensions=DIMENSIONS,seasons=sorted(SEASONS),number_words=NUMBER_WORDS))
    summary={}
    for domain in ('Toys','Beauty'):
        p=RAW/domain/'catalog.json';prior=PREVIOUS/(domain+'_relations.json')
        for source in (p,prior):manifest[str(source.relative_to(ROOT))]=sha(source)
        assert manifest[str(p.relative_to(ROOT))]==json.loads((RAW/'manifest.json').read_text())['sha256'][str(p.relative_to(ROOT))]
        assert manifest[str(prior.relative_to(ROOT))]==json.loads((PREVIOUS/'verification_manifest.json').read_text())[str(prior.relative_to(ROOT))]
        engine=TypedRelations(json.loads(p.read_text()),json.loads(prior.read_text()),domain);result=engine.export()
        write(OUT/(domain+'_relations.json'),result);manifest[str((OUT/(domain+'_relations.json')).relative_to(ROOT))]=sha(OUT/(domain+'_relations.json'))
        summary[domain]=result['coverage'];print(domain,result['coverage'],flush=True)
    write(OUT/'manifest.json',manifest)
    write(OUT/'extraction_summary.json',dict(domains=summary,seconds=time.monotonic()-started,outcome_files_read=False))
    write(OUT/'status.json',dict(state='AWAITING_SOURCE_AUDIT',outcome_files_read=False,gpu_inferences=0,semantic_training_state='USER_PAUSED'))


if __name__=='__main__':main()
