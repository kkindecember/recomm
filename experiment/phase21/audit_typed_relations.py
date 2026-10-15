"""Persist source review before opening this round's recommendation outcomes."""
import json
from pathlib import Path
from analyze_selective_evidence import sha,write
from extract_typed_relations import ROOT,OUT


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for p,h in manifest.items():assert sha(ROOT/p)==h
    reviews={};quality={}
    for domain in ('Toys','Beauty'):
        data=json.loads((OUT/(domain+'_relations.json')).read_text());identity=[];version=[];changes=[]
        # Every displayed title was inspected. Support means only the literal parent identity.
        for c in data['identity_audit']:
            r=c['record'];identity.append(dict(audit_id=c['audit_id'],item=r['item'],verdict='supported',
                note='主体名称位于边界之前，源标题支持此父家族／主题；不证明同功能或同生产方。'))
        for c in data['version_audit']:
            r=c['record'];version.append(dict(audit_id=c['audit_id'],item=r['item'],verdict='supported',
                fields={k:v['value'] for k,v in r['versions'].items() if v['status']=='known'},
                note='所列字段均在主体标题明确出现；批次是series编号，year仅是套系／季节年度，缺失字段未补猜。'))
        for r in data['role_changes']:
            assert domain=='Beauty' and r['item'] in (4145,4907,9993)
            verdict='supported' if r['item']==9993 else 'unsupported'
            note=('烘干灯提及适用OPI GelColor，删除自身属于该家族的断言正确；未把提及自动当兼容边。'
                  if r['item']==9993 else 'works是Bath Body Works名称的一部分；单词边界启发式错误截断身份，产生漏识别。')
            changes.append(dict(item=r['item'],verdict=verdict,note=note))
        # Include all changed-role decisions, not only retained positive identities, in the gate.
        unique={r['item']:r for r in identity}
        unique.update({r['item']:r for r in changes})
        gates={}
        for label,rows in [('identity',list(unique.values())),('version',version)]:
            supported=sum(r['verdict']=='supported' for r in rows)
            gates[label]=dict(units=len(rows),supported=supported,unsupported=len(rows)-supported,
                fraction=supported/len(rows) if rows else None,passed=len(rows)>=10 and supported/len(rows)>=.9)
        gates['passed']=all(gates[k]['passed'] for k in ('identity','version'));quality[domain]=gates
        reviews[domain]=dict(identity=identity,version=version,role_changes=changes)
    write(OUT/'source_review.json',dict(reviewer='same assistant; no independent semantic annotation',outcome_files_read=False,
        identity_gate_units='union of SHA identity audit and all role changes, deduplicated by item',domains=reviews))
    write(OUT/'quality_gate.json',dict(domains=quality,eligible_new_policies=all(r['passed'] for r in quality.values()),
        compatibility_edges_eligible=False,known_errors_retained=True,threshold='engineering screening, not population precision'))
    for p in [Path(__file__),OUT/'source_review.json',OUT/'quality_gate.json']:manifest[str(p.relative_to(ROOT))]=sha(p)
    write(OUT/'quality_manifest.json',manifest)
    write(OUT/'status.json',dict(state='SOURCE_AUDIT_COMPLETED',outcome_files_read=False,quality=quality,semantic_training_state='USER_PAUSED'))
    print(json.dumps(quality),flush=True)


if __name__=='__main__':main()
