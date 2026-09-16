"""仅供评估的共同 mask 比较；在 analysis 环境中运行。"""
from pathlib import Path
import sys, json, csv, hashlib
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path('/mnt/ssd/zliu/phase_restart')
sys.path.insert(0, str(ROOT))
from pr.gate import EvalGate, sha256_file
from pr import refeval
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'test_res/020-20260913_071841-v1-p9016-joint'
SOFT = Path('/work/phase3/test_res/124-20260905_185119-p9016_step_policy_1m/native/fixed_seed124101/softall/p9016_full.coords.tsv')
V1 = RUN / 'selected.3dg'
selection = json.loads((RUN/'selection.json').read_text())
assert selection['selection']['selected_id'] == 'random_joint'
assert sha256_file(V1) == selection['selection']['selected_coordinate']['sha256']
assert sha256_file(SOFT) == 'b5366dcdcb5faaba25ad7ec1cf1ecd0de3e920982db233b1a4b20ee346605704'
assert not (OUT/'summary.json').exists(), 'Do not overwrite completed analysis'
(OUT/'plots').mkdir(exist_ok=True)
gate = EvalGate(str(OUT/'gate.json'))
for label, path in [('softall', SOFT), ('V1', V1)]:
    gate.register(refeval.STAGE, label, str(path))
config = json.loads((RUN/'config.json').read_text())
chroms = [v['name'] for v in config['coordinate_grid']['chromosomes']]
lengths = {v['name']:v['length_bp'] for v in config['coordinate_grid']['chromosomes']}
soft, v1 = {}, {}
with SOFT.open() as f:
    for r in csv.DictReader(f, delimiter='\t'):
        key = (r['chr'], int(r['copy']))
        d = soft.setdefault(key,{})
        p = int(r['start'])
        assert p not in d and p % 1000000 == 0
        d[p] = np.array([float(r[a]) for a in 'xyz'])
with V1.open() as f:
    for line in f:
        a = line.split()
        if not a or a[0].startswith('#'): continue
        key = (chroms[int(a[0][1:3])-1], 'ab'.index(a[0][3]))
        d = v1.setdefault(key,{})
        p = int(a[1])
        assert p not in d and p % 1000000 == 0
        d[p] = np.array(list(map(float,a[2:5])))
assert len(soft) == len(v1) == 40
gate.arm(refeval.STAGE)
ref = refeval.load_reference(gate)
refhash = sha256_file(ROOT/'data/P9016.1m.3dg.gz')
assert refhash == '1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29'
rows, copies, masks = [], [], {}
for chrom in chroms:
    tracks = [ref[f'{chrom}(mat)'], ref[f'{chrom}(pat)'], soft[(chrom,0)], soft[(chrom,1)], v1[(chrom,0)], v1[(chrom,1)]]
    pos = sorted(p for p in set.intersection(*(set(t) for t in tracks)) if 3000000 <= p < lengths[chrom] and all(np.isfinite(t[p]).all() for t in tracks))
    assert len(pos) >= 4 and all(p % 1000000 == 0 for p in pos)
    masks[chrom] = pos
    distances = [pdist(np.array([t[p] for p in pos])) for t in tracks]
    n = len(pos)*(len(pos)-1)//2
    assert all(len(d)==n and np.isfinite(d).all() for d in distances)
    for name, offset in [('Softall',2),('V1 selected',4)]:
        rho = np.array([[spearmanr(distances[offset+i], distances[j]).statistic for j in range(2)] for i in range(2)])
        assert np.isfinite(rho).all()
        direct, swapped = np.trace(rho)/2, (rho[0,1]+rho[1,0])/2
        swap = bool(swapped > direct)
        tied = bool(abs(direct-swapped) <= 1e-12)
        vals = [float(rho[i,1-i if swap else i]) for i in range(2)]
        row = dict(chromosome=chrom,method=name,n_bins=len(pos),n_pairs=n,orientation='tie' if tied else ('swapped' if swap else 'direct'),rho_a_mat=rho[0,0],rho_a_pat=rho[0,1],rho_b_mat=rho[1,0],rho_b_pat=rho[1,1],matched=float(max(direct,swapped)),cross_matched=float(min(direct,swapped)),contrast=float(abs(direct-swapped)))
        rows.append(row)
        for i,val in enumerate(vals): copies.append(dict(chromosome=chrom,method=name,copy='ab'[i],reference_copy=('pat' if (1-i if swap else i) else 'mat'),spearman=val,n_pairs=n,orientation_tied=tied))
def tsv(name, records):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),delimiter='\t'); w.writeheader(); w.writerows(records)
tsv('per_chromosome.tsv',rows); tsv('per_copy.tsv',copies)
(OUT/'common_positions.json').write_text(json.dumps(masks,indent=2))
summary = {}
for name in ['Softall','V1 selected']:
    rr = [r for r in rows if r['method']==name]
    summary[name] = {key:dict(mean=float(np.mean([r[key] for r in rr])),median=float(np.median([r[key] for r in rr]))) for key in ['matched','cross_matched','contrast']}
a = np.array([r['matched'] for r in rows if r['method']=='Softall'])
b = np.array([r['matched'] for r in rows if r['method']=='V1 selected'])
summary['comparison'] = dict(v1_minus_softall_mean=float(np.mean(b-a)),v1_wins=int(sum(b>a)),ties=int(sum(b==a)),softall_wins=int(sum(b<a)),chromosomes=20,common_pairs=sum(r['n_pairs'] for r in rows if r['method']=='Softall'))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
fig, axes = plt.subplots(1,2,figsize=(6,3))
for ax,key,title in zip(axes,['matched','contrast'],['Similarity to reference','Copy-specific R2 contrast']):
    data = [np.array([r[key] for r in rows if r['method']==name]) for name in ['Softall','V1 selected']]
    boxes = ax.boxplot(data,positions=[1,2],widths=.45,patch_artist=True,showfliers=False,medianprops={'color':'black','linewidth':1})
    for patch,color in zip(boxes['boxes'],['#65a5cc','#edaa55']): patch.set_facecolor(color); patch.set_alpha(.55)
    for i in range(20):
        ax.plot([1,2],[d[i] for d in data],color='#aaaaaa',alpha=.5,lw=.5,zorder=1)
    for x,d,color in zip([1,2],data,['#2677a6','#bd650a']): ax.scatter(np.full(20,x),d,s=9,color=color,zorder=3)
    ax.set_xticks([1,2],['Softall\nseed124101','V1 selected\nrandom_joint'])
    ax.set_ylabel('Mean matched Spearman rho' if key=='matched' else 'Matched minus cross-matched rho')
    ax.set_title(title); ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(min(0,min(map(np.min,data))-.03),max(map(np.max,data))+.05)
fig.suptitle('P9016 | same bin pairs, 20 chromosomes',fontsize=7)
fig.text(.5,.015,'Each dot = one chromosome (mean of two copies); lines connect the same chromosome.',ha='center',fontsize=7)
fig.tight_layout(rect=[0,.05,1,.95])
for ext in ['png','pdf']: fig.savefig(OUT/'plots'/('softall-v1-reference-spearman.'+ext),dpi=300)
plt.close(fig)
provenance = dict(inputs={str(p):sha256_file(p) for p in [SOFT,V1,ROOT/'data/P9016.1m.3dg.gz']},script_sha256=sha256_file(__file__),reference_read_after_coordinate_hash=True,phase_pairs_read=False,refitting=False,selection_changed=False)
(OUT/'config.json').write_text(json.dumps(dict(method='Spearman of Euclidean intra-track upper-triangle distances',grid='Exact shared 1 Mb bin starts, >=3 Mb, below header length; common finite mask across six tracks',gauge='One geometry-selected whole-chromosome swap per method, maximizing mean rho; no local swaps',softall_seed=124101,v1_candidate='random_joint',biological_samples=1,provenance=provenance),indent=2))
text = '# Softall 与 V1：相对参考结构的 Spearman 相关\n\n'
text += '评价已有坐标，没有重拟合或重新选择候选。Softall 固定 seed124101；V1 为 020 预先选中的 random_joint。\n\n'
text += '每条染色体在六条轨迹共同有效的相同基因组位置上计算非对角距离。使用精确 1 Mb bin 起点对齐，保留 >=3 Mb 且小于 header 长度的位置。不插值补点。每种方法独立允许整条染色体交换 A/B，使两 copy 的平均 Spearman 相关最大；同时保存四种相关及交换对照。\n\n'
text += '| 方法 | matched 平均 | matched 中位数 | cross-matched 平均 | R2 contrast 平均 |\n|---|---:|---:|---:|---:|\n'
for name in ['Softall','V1 selected']:
    s=summary[name];text+=f"| {name} | {s['matched']['mean']:.6f} | {s['matched']['median']:.6f} | {s['cross_matched']['mean']:.6f} | {s['contrast']['mean']:.6f} |\n"
text+='\n箱线图每个点是一条染色体的两 copy 平均（每组 20 点），配对线连接同一条染色体。箱体为四分位距，中线为中位数，须为 1.5 IQR 范围内数据；所有点均显示。每 copy 的 40 个相关见 per_copy.tsv。\n\n'
text+='共享染色质形状及基因组距离趋势可提高 matched 相关；它不单独证明等位拷贝恢复。最佳整体配对也会产生正向选择效应，contrast 不应与零直接作显著性检验。这里只报告描述性比较；20 条染色体来自同一个细胞，不是独立生物学重复。共同 mask 与 canonical 020 指标的比较集合不同，故数值可能不同。\n'
(OUT/'README.md').write_text(text)
print(json.dumps(summary,indent=2))
print('chr1:',json.dumps([r for r in rows if r['chromosome']=='chr1']))
