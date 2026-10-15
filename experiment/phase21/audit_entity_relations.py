"""Record source-only assistant review; no recommendation outcomes are loaded."""
import json
from pathlib import Path
from analyze_selective_evidence import sha, write
from extract_entity_relations import ROOT, OUT, RAW
from analyze_history_evidence import fields

# Reviewed at the preregistered PARENT family/theme level, never same batch/function.
COMPAT_NOTES = {
    '0016b0cbe3efd5d1': 'other不是产品名称，目录中碰巧出现该词。',
    '07764cd3ca30608b': '明确not for use，否定保留正确。',
    '0154225995356d57': '句子指向主机，唯一命中却是同样提到主机的电源适配器。',
    '23bbfdb4f2d44ecb': 'learning丢失品牌和系统限定，过宽结果已隔离。',
    '030a3511bb59b168': '没有查到不等于没有兼容关系。',
    '01253b782f1464da': '命中的多为也适配该主机的配件，不是被适配主机。',
    '0a222b263a86e52c': 'not included修饰耳机是否随附，不能否定随后compatible；旧否定窗口误伤。',
    '06fd7a7d84f96b3f': '重复works with被当作别名一部分，且把两个配件互相连接。',
    '2f2b93ba54bb30fa': '只有品牌没有具体套系，过宽结果已隔离。',
    '087531678a6cb734': 'tag未通过单词长度门槛，保留未匹配；不能推断无关系。',
    '0135e7a06450abe1': 'popular是普通形容词，不能据此认定Popular Playthings产品。',
    '0d8d8e4574fc4c6b': '明确not for use，否定保留正确。',
    '07c04515f1e85d5f': '目标标题compare to Rainbow Loom是比较性提及，不能确认为该产品。',
    '501f57a737e3c069': 'thomas宽主题命中，且原amp分号截断，不能作为准确配套对象。',
    '0923a7fb461e0366': '普通home computer指代，没有命中。',
    '023d2333b389e74a': 'learning被保留但Explorer限定丢失，旧Leappad和配件混入。',
    '081d6c17efa5715d': 'gel/acrylic是材料类别，命中装饰和工具并非确认的配套对象。',
    '1f11edbcc0ec3120': '明确not a replacement，否定保留正确。',
    '064ba2f7fe57217d': 'rhinestones/glitters指普通装饰类别；唯一商品命中不构成具体型号确认。',
    '0387b4cba7a2d986': 'works with your skin指皮肤，不是目录对象。',
    '0b153011e3b98ebe': 'your body指身体，不能连接标题碰巧含该词的商品。',
    '46cf5968ceba5f0b': '明确not as a replacement，否定保留正确。',
    '125d2f370fb040e8': 'types of hair指发质，被错误连接到脱发补充剂。',
    '0404436a80f2f847': '指真指甲和假指甲，没有解析为具体商品。',
    '0bc317fb612a8b23': 'gel/acrylic是材料类别，命中装饰和工具并非确认的配套对象。',
    'e3c1a18d07f6dc47': '明确not as a replacement，否定保留正确。',
    '24c692a08f818179': 'your skins指皮肤，被错误连接到另一种皮肤护理品。',
    '05aa338aca1699f4': '泛指热造型工具，没有命中不表示没有关系。',
    '1967e0b5592a0c31': 'humidity指环境湿度，不是产品实体。',
    '29f37f7595656b2c': 'rhinestones/glitters指普通装饰类别；唯一商品命中不构成具体型号确认。',
    '0dcbcecdbbf897eb': '泛指热造型工具，没有命中不表示没有关系。',
    '22418d68f46aa7ea': 'supersoft是形容词，丢失scrublet对象，不能连接软海绵或其他商品。',
}


def main():
    manifest = json.loads((OUT/'manifest.json').read_text())
    for p, digest in manifest.items():assert sha(ROOT/p) == digest
    domains = {}; reviews = {};regressions = {}
    for domain in ('Toys', 'Beauty'):
        data = json.loads((OUT/(domain+'_relations.json')).read_text())
        cards = []
        for r in data['family_audit']:
            item = r['item'];failed = domain == 'Beauty' and item['item'] == 9993
            note = ('标题和种子支持同一命名家族或主题；不证明同品牌、同批次、同功能或可替代。')
            if failed:note = '烘干灯只是列举适配的OPI GelColor，商品本身不属于该指甲油家族。'
            if domain == 'Beauty' and item['item'] in (11191, 11192):
                note = '只支持Essie夏季发布主题这个父层级：2013不同于2012，不能当成同年度套系；当前batch也没有记录collection年份。'
            cards.append(dict(audit_id=r['audit_id'], item=item['item'], verdict='unsupported' if failed else 'supported', note=note))
        compat = []
        for r in data['compatibility_audit']:
            note = COMPAT_NOTES[r['audit_id']]
            verdict = ('correctly_withheld' if r['status'] in ('unmatched','too_broad','negated') else 'not_a_confirmed_item_relation')
            if r['audit_id'] == '0a222b263a86e52c':verdict = 'incorrect_negation'
            compat.append(dict(audit_id=r['audit_id'], status=r['status'], verdict=verdict, note=note))
        supported = sum(r['verdict'] == 'supported' for r in cards)
        domains[domain] = dict(reviewed=len(cards), supported=supported, unsupported=len(cards)-supported,
            supported_fraction=supported/len(cards), passed=len(cards)>=10 and supported/len(cards)>=.9)
        reviews[domain] = dict(family_alias=cards, compatibility=compat)
        catalog = json.loads((RAW/domain/'catalog.json').read_text())
        ids = ([11649,11356,10482,10095,10098] if domain=='Toys' else [3034,5662,8650,8421,11224,1159,11731,4340,9993,11191,11192])
        # Source-only regression cards; intentionally no targets, user IDs or gains/losses.
        regressions[domain] = [dict(item=i, source=fields(catalog['passages'][i]),
            metadata=data['items'][i-1], compatibility=[r for r in data['compatibility_lookups'] if r['source']==i]) for i in ids]
        if domain == 'Beauty':
            for r in data['items']:
                if r['size_after_series']:
                    regressions[domain].append(dict(item=r['item'],source=fields(catalog['passages'][r['item']]),metadata=r))
    write(OUT/'source_review.json', dict(reviewer='same assistant; source-only review, not independent annotation',
        criterion='named parent family/theme only, not same batch, year, manufacturer or function', domains=reviews))
    write(OUT/'source_regression_cases.json', regressions)
    write(OUT/'quality_gate.json', dict(domains=domains, eligible_new_policies=all(r['passed'] for r in domains.values()),
        compatibility_edges_eligible=False, limitation='engineering threshold only; one known false positive is retained, not cleaned after audit'))
    for p in [Path(__file__),OUT/'source_review.json',OUT/'quality_gate.json',OUT/'source_regression_cases.json']:
        manifest[str(p.relative_to(ROOT))] = sha(p)
    write(OUT/'quality_manifest.json', manifest)
    write(OUT/'status.json', dict(state='SOURCE_AUDIT_COMPLETED', outcome_files_read=False, quality=domains))
    print(json.dumps(domains), flush=True)


if __name__ == '__main__':main()
