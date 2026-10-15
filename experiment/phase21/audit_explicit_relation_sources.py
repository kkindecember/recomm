"""Persist source-only judgments made on the fixed blinded catalogue cards."""
import json
from pathlib import Path

from analyze_selective_evidence import sha,write

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/explicit_relations_20260925_v1'

# Reviewed from source cards and, for ambiguous series entries, complete catalogue descriptions.
# These labels are for quality assessment only; no edge/group is deleted from evaluation.
UNCERTAIN={
    '7279cdeb701eab56':'The New York Doll Collection前缀同时覆盖购物车和推车，现有字段不足以区分品牌称呼与具体系列。',
    'a3924e189d5bf910':'sticker collection在描述中也只是“一套贴纸”的普通含义，不能确定是命名系列。',
    '3754d5a5f60d7d7e':'mini collection可指小规格指甲油套装，描述未建立独立命名系列，系列实体边界不确定。',
    '3e91e0cabc9546b0':'提取把1 inch中的1当成系列编号；源标题支持Pro Series，但不支持提取出的Series 1名称。',
    '442e8913807d2382':'John Frieda Collection前缀不足以排除仅为品牌表达；更具体的Frizz-Ease在collection之后，未被键捕获。',
    '54c119822a523e24':'compatible with的直接宾语是all loom kits；Rainbow Loom在后续营销话语出现，直接商品链接范围不确定。',
    'b43c54a613dc2aa9':'源句明确提到Europe地图，提取却链接较短的Ticket to Ride标题；版本适配不能仅凭前缀确认。',
}
EXTRA_SOURCE_NOTES={
    '6448b4e1077c1519':'已补读6228和8302的description，两者均明示from bdellium maestro series；8302标题中的studio line不据此推翻描述中的系列归属。',
    'b5742bd673816f94':'完整源句说明Alchemy需Dominion或独立扩展，并列举Intrigue；支持组合使用，但不能仅把局部expansion to视作主谓指向。',
    '0b00b3938f6ce8a8':'共同命名对象是Star Trek The Original Series主题，支持同主题产品线，不代表同一功能或同一配套需求。',
}


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for rel,digest in manifest.items():assert sha(ROOT/rel)==digest
    labels=[];quality={}
    for d in ('Toys','Beauty'):
        cards=json.loads((OUT/(d+'_relations.json')).read_text())['blind_audit_cards'];quality[d]={}
        for card in cards:
            aid=card['audit_id'];ok=aid not in UNCERTAIN
            default=('展示标题均明示相同命名主题／系列标签，非仅共享一般类别词。' if card['kind']=='series'
                     else '完整源句或扩展标题明示候选与所匹配商品的适配／扩展关联；不推断下一次购买。')
            labels.append(dict(audit_id=aid,domain=d,kind=card['kind'],judgment='supported' if ok else 'uncertain',
                rationale=UNCERTAIN.get(aid,EXTRA_SOURCE_NOTES.get(aid,default))))
        for kind in ('series','explicit'):
            rows=[r for r in labels if r['domain']==d and r['kind']==kind];n=len(rows)
            supported=sum(r['judgment']=='supported' for r in rows)
            quality[d][kind]=dict(audited=n,supported=supported,uncertain=n-supported,
                supported_fraction=supported/n if n else None,gate_pass=n>=10 and supported/n>=.9)
    assert set(UNCERTAIN)<=set(r['audit_id'] for r in labels)
    passed=[kind for kind in ('series','explicit') if all(quality[d][kind]['gate_pass'] for d in quality)]
    write(OUT/'source_audit_labels.json',dict(reviewer='same assistant session; source-only inspection, not independent human annotation',
        outcome_files_read_for_judgment=False,labels=labels))
    write(OUT/'quality_gate.json',dict(domains=quality,eligible_relation_types=passed,
        thresholds=dict(min_audited=10,min_supported_fraction=.9),threshold_is_engineering_not_statistical=True,
        manually_uncertain_relations_not_removed=True,full_catalog_semantic_accuracy_not_estimated=True))
    write(OUT/'quality_audit_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in
        [Path(__file__),OUT/'source_audit_labels.json',OUT/'quality_gate.json',OUT/'Toys_relations.json',OUT/'Beauty_relations.json']})
    print(json.dumps(dict(quality=quality,eligible=passed),ensure_ascii=False))


if __name__=='__main__':main()
