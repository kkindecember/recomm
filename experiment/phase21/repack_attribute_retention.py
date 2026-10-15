"""Pre-GPU correction of unused description budget; keeps the rejected preparation."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from transformers import AutoTokenizer
from experiment.phase21.attribute_retention import ROOT, OUT, PLAN, SOURCE, ARMS, read, write, sha, stamp
from experiment.phase21.attribute_retention_core import parse_fields, EMPTY, pack_catalog
from experiment.phase20.lime_clean_gram_core import T5


def main():
    archive=OUT.with_name('attribute_retention_20260923_preflight_budget40')
    assert read(OUT/'status.json')['state']=='PREPARED' and not archive.exists()
    assert not any(OUT.glob('*/worker.log'))
    OUT.rename(archive)
    shutil.copytree(archive,OUT)
    write(OUT/'status.json',dict(state='REPACKING',updated_at=stamp(),reason='Fill unused token budget before GPU work'))
    tokenizer=AutoTokenizer.from_pretrained(str(T5),local_files_only=True)
    for domain in ('Toys','Beauty'):
        folder=OUT/domain; catalog=read(SOURCE/domain/'catalog.json'); audit=read(folder/'packing_audit.json')
        old=dict(np.load(folder/'tokens_retained.npz')); size=len(catalog['items'])
        bank=np.zeros_like(old['tokens']);lengths=np.zeros_like(old['lengths']);groups={}; offsets={}
        fields=[None]+[parse_fields(p) for p in catalog['passages'][1:]]
        for start in range(1,size,64):
            stop=min(start+64,size)
            strings=[' '+fields[i].get('description','na') for i in range(start,stop)]
            encoded=tokenizer(strings,add_special_tokens=False,truncation=True,max_length=999)['input_ids']
            for i,raw in zip(range(start,stop),encoded):
                a=audit['description_offsets'][str(i)][0]
                content=[v for v in raw if v not in (0,1,1820,9175)][:127-a]
                bank[i,:a]=old['tokens'][i,:a];bank[i,a:a+len(content)]=content
                n=a+len(content)+1;bank[i,n-1]=1;lengths[i]=n;offsets[str(i)]=[a,n-1]
                if fields[i].get('description','na').strip().lower() not in EMPTY and content:
                    groups.setdefault(len(content),[]).append(i)
        shuffled=bank.copy();donors=np.arange(size,dtype=np.int64)
        for width,items in groups.items():
            items.sort(key=lambda i:hashlib.sha256(f'20260923|description-placebo|{domain}|{catalog["items"][i]}'.encode()).digest())
            if len(items)<2:continue
            for receiver,donor in zip(items,items[1:]+items[:1]):
                a,b=offsets[str(receiver)];c,d=offsets[str(donor)]
                assert b-a==d-c==width
                shuffled[receiver,a:b]=bank[donor,c:d];donors[receiver]=donor
        changed=0
        for i in range(1,size):
            a,b=offsets[str(i)]
            np.testing.assert_array_equal(bank[i,:a],shuffled[i,:a]);np.testing.assert_array_equal(bank[i,b:],shuffled[i,b:])
            assert bank[i,lengths[i]-1]==shuffled[i,lengths[i]-1]==1 and lengths[i]<=128
            changed+=int(not np.array_equal(bank[i],shuffled[i]))
        # Full core reconstruction on a deterministic catalog subset, including its own donor mapping.
        selected=list(range(1,65))
        small={key:['']+[catalog[key][i] for i in selected] for key in ('items','lexical_ids','passages')}
        actual,lens,_,_=pack_catalog(small,tokenizer,domain)
        np.testing.assert_array_equal(actual['retained'][1:],bank[selected]);np.testing.assert_array_equal(lens['retained'][1:],lengths[selected])
        assert changed>=1000
        np.savez_compressed(folder/'tokens_retained.npz',tokens=bank,lengths=lengths)
        np.savez_compressed(folder/'tokens_shuffled.npz',tokens=shuffled,lengths=lengths)
        np.save(folder/'description_donors.npy',donors)
        audit.update(description_offsets=offsets,placebo_changed_items=changed,
                     retained_full_budget_items=int((lengths[1:]==128).sum()),
                     pre_gpu_budget_revised=True,revised_core_parity_items=selected)
        audit['mean_tokens']['retained']=audit['mean_tokens']['shuffled']=float(lengths[1:].mean())
        audit['examples']=[dict(item_id=i,original=tokenizer.decode(np.load(folder/'tokens_original.npz')['tokens'][i],skip_special_tokens=True),
                               retained=tokenizer.decode(bank[i],skip_special_tokens=True)) for i in (1,2,3)]
        write(folder/'packing_audit.json',audit)
        manifest=read(folder/'input_manifest.json')
        for p in list(folder.glob('tokens_*.npz'))+[folder/'description_donors.npy']:
            manifest['sha256'][str(p.relative_to(ROOT))]=sha(p)
        write(folder/'input_manifest.json',manifest)
        state=read(folder/'status.json');state.update(updated_at=stamp(),placebo_changed_items=changed,
               description_budget='all remaining tokens up to 128 total');write(folder/'status.json',state)
        print(json.dumps(dict(domain=domain,mean_tokens=audit['mean_tokens'],changed=changed)),flush=True)
    manifest=read(OUT/'source_manifest.json')
    for relative in list(manifest['sha256']):
        p=ROOT/relative;manifest['sha256'][relative]=sha(p);shutil.copy2(p,OUT/'sources'/relative)
    for p in (Path(__file__),ROOT/'experiment/phase21/revise_attribute_retention.sh'):
        manifest['sha256'][str(p.relative_to(ROOT))]=sha(p)
        dest=OUT/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    write(OUT/'source_manifest.json',manifest);shutil.copy2(PLAN,OUT/'protocol.md')
    write(OUT/'budget_revision.json',dict(state='VERIFIED',created_at=stamp(),archive=str(archive.relative_to(ROOT)),
          new_training=False,new_generation=False,development_effectiveness_observed=False,core_reconstruction_exact=True))
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp(),revision='Remaining description token budget',
          domains={d:read(OUT/d/'status.json') for d in ('Toys','Beauty')}))


if __name__=='__main__':main()
