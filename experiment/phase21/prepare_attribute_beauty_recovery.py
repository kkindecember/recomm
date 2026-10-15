"""Reuse frozen inputs verbatim after a resource-only stop; no science changes."""
from pathlib import Path
import shutil
from experiment.phase21.attribute_retention import ROOT, read, write, sha, stamp


def main():
    old=ROOT/'artifacts/phase21/attribute_retention_20260923_v1'
    out=ROOT/'artifacts/phase21/attribute_retention_beauty_recovery_20260923_v1'
    assert read(old/'status.json')['state']=='RESOURCE_BUDGET_STOP'
    out.mkdir(exist_ok=False);folder=out/'Beauty';folder.mkdir()
    names=['samples.json','data_config.json','packing_audit.json','description_donors.npy']
    names += [f'tokens_{arm}.npz' for arm in ('original','retained','shuffled')]
    for name in names:
        shutil.copy2(old/'Beauty'/name,folder/name)
        assert sha(old/'Beauty'/name)==sha(folder/name)
    manifest=read(old/'Beauty/input_manifest.json')
    replaced={}
    for relative,digest in manifest['sha256'].items():
        source=ROOT/relative
        destination=folder/source.name if source.parent==old/'Beauty' else source
        replaced[str(destination.relative_to(ROOT))]=digest
        if destination.parent==folder: assert sha(destination)==digest
    write(folder/'input_manifest.json',dict(manifest,sha256=replaced))
    source_paths=[ROOT/relative for relative in read(old/'source_manifest.json')['sha256']]
    plan=ROOT/'plan/第二十一阶段/21.1B_Beauty单域资源恢复_20260923.md'
    source_paths += [Path(__file__),ROOT/'experiment/phase21/run_attribute_beauty_recovery.sh',plan]
    source_manifest={}
    for path in source_paths:
        relative=str(path.relative_to(ROOT));source_manifest[relative]=sha(path)
        destination=out/'sources'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    write(out/'source_manifest.json',dict(sha256=source_manifest))
    shutil.copy2(plan,out/'protocol.md')
    write(out/'reuse_receipt.json',dict(created_at=stamp(),prior_attempt=str(old.relative_to(ROOT)),
        reused_files_exact=names,science_changes=[],scheduled_domains=['Beauty'],
        reason='Resource state changed; prioritize the domain with the strongest truncation evidence',
        test_read=False,development_outcomes_used_for_recovery=False))
    write(folder/'status.json',dict(state='PREPARED',updated_at=stamp(),domain='Beauty'))
    write(out/'status.json',dict(state='PREPARED',updated_at=stamp(),scheduled_domains=['Beauty']))
    print(str(out))


if __name__=='__main__':main()
