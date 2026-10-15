"""Freeze catalogue extraction and outcome-blind review materials before evaluation."""
import json
import time
from pathlib import Path
from analyze_selective_evidence import sha, write
from entity_relation_core import EntityRelations

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/phase21/entity_relations_20260925_v1'
RAW = ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
OLD = ROOT/'artifacts/phase21/explicit_relations_20260925_v1'
PROTOCOL = ROOT/'plan/第二十一阶段/21.15_实体归一化与未知状态固定检查_20260925.md'


def main():
    started = time.monotonic();OUT.mkdir(exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name('entity_relation_core.py'),
             Path(__file__).with_name('explicit_relation_core.py'),
             Path(__file__).with_name('analyze_history_evidence.py'), PROTOCOL]
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for p in paths:
        (OUT/p.name).write_text(p.read_text())
    summary = {}
    for domain in ('Toys', 'Beauty'):
        p = RAW/domain/'catalog.json';old = OLD/(domain+'_relations.json')
        manifest[str(p.relative_to(ROOT))] = sha(p);manifest[str(old.relative_to(ROOT))] = sha(old)
        assert manifest[str(p.relative_to(ROOT))] == json.loads((RAW/'manifest.json').read_text())['sha256'][str(p.relative_to(ROOT))]
        engine = EntityRelations(json.loads(p.read_text()), json.loads(old.read_text()), domain)
        result = engine.export();write(OUT/(domain+'_relations.json'), result)
        manifest[str((OUT/(domain+'_relations.json')).relative_to(ROOT))] = sha(OUT/(domain+'_relations.json'))
        summary[domain] = result['coverage'];print(domain, result['coverage'], flush=True)
    write(OUT/'extraction_summary.json', dict(domains=summary, seconds=time.monotonic()-started, outcome_files_read=False))
    write(OUT/'manifest.json', manifest)
    write(OUT/'status.json', dict(state='AWAITING_SOURCE_AUDIT', outcome_files_read=False, gpu_inferences=0))


if __name__ == '__main__':main()
