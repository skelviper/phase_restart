# 融合 CUDA 内核设计

## 范围

该目录是实验 028 的独立实现面。它只读取 `SparseAggregate` 表示的七列 SNP-free 聚合数据。不打开 phase 字段、参考坐标或正式 020 运行器。归档的 020 源码仍是数学权威。

实现目标是 20 kb 完整网格：131,700 个基因组位点、263,400 个物理珠子以及 8,672,379,150 个无序可用位点对。全部 40 条拷贝轨迹和全部 1,703,888 条原始记录都保留在范围内。不允许配对抽样、基因组子集化、稠密配对数组或降低优化器预算。

## 数学契约

对于可用位点对 `(i,j)`，定义 `A=e_i e_j`、`S=K(X_i,X_j)+K(Y_i,Y_j)`，以及 `C=K(X_i,Y_j)+K(Y_i,X_j)`。精确速率为：

```text
cis:   r_ij = A [p S + (1-p) C] / 2
inter: r_ij = A [S + C] / 4
```

其中 `K=1e-6 + (1-1e-6)*(1+||delta||^2/(2*l0)^2)^-2`，`e_i=sqrt(endpoint_count_i+10)/full_grid_mean`。原始同区间记录不是可用的结构配对，只由候选无关的按 bin 饱和层表示。

对于 `{cis_offdiag, inter}` 中的 `g`，`M_g` 是原始分组总数，`R_g=sum_(i,j in E_g) r_ij`。计数值为：

```text
sum_g [M_g*log(R_g) - sum_(observed ij in g) C_ij*log(r_ij)] + NLL_diag
```

其中条件阶乘常数会被记录但省略，同区间阶乘项计入 `NLL_diag`。完整计数值及其 x/p 梯度按 `M_all=1,703,888` 归一化。

坐标分解为：

```text
g_x,count = [(M_cis/R_cis)*grad(R_cis)
             +(M_inter/R_inter)*grad(R_inter)
             - sum_observed C_ij/r_ij * grad(r_ij)] / M_all

g_p,count = [(M_cis/R_cis)*dR_cis/dp
             - sum_observed_cis C_ij/r_ij * dr_ij/dp] / M_all
```

`dr/dp = e_i e_j*(S-C)/2` 对 cis 成立，inter 为零。无约束的 q 梯度为 `dp/dq*(g_p,count + d(p_prior)/dp)`。实际执行的排斥半径是 `0.7*l0`，不是过时的 020 元数据值 `2.0*l0`。

## 内核归属与归约

`full_ordered_rows_kernel` 为每个 CUDA block 分配一个中心位点，并扫描所有 `j != i`。因此每条无序边会以两个方向访问。内核在每个方向计算四种拷贝组合，并将接触 K/导数与全部四项跨位点排斥融合。每个 block 只写入其中心位点梯度，因此不使用全局坐标原子操作。

完整行的标量输出对重复的有序访问使用二分之一因子：`R_cis`、`R_inter`、`dR_cis/dp` 和跨位点排斥能量都适用。中心坐标梯度不使用该因子，因为每个端点恰好作为中心一次。同源排斥对每个位点添加一次。总排斥为 `(cross/2 + homolog)/(2*N)`，其坐标梯度为中心行梯度除以 `2*N`。

每行使用固定的 warp-shuffle 加共享内存树归约；第二棵固定树归约 O(N) 行标量。这样，同一设备/构建上的重复评估无需依赖 float64 全局原子操作即可复现。求和顺序仍可能不同于归档 NumPy，比较时使用预先声明的容差。

`sparse_observed_rows_kernel` 接收正的观测上三角边的两个 CSR 视图。输出视图为观测对数速率和 p 导数各计每条边一次。输出加输入视图在每个端点各计算一次中心坐标梯度，不使用原子操作。CSR 构造基于已有排序三角索引，复杂度为 O(N+M)；需要时三角运算和整数计数保留为 int64。

距离为零时，接触 K 有限，其导数为零向量。排斥和 bond 风格的径向导数在距离为零时显式选择零向量，同时保留有限的 hinge 值。不允许除零到达结果。

## Python 接口与构建

`fused_objective.py` 暴露 `FusedObjective`（028 `TorchObjective` 接口的直接替代子类）以及 `load_fused_extension`。它保持 sphere pullback、bond、bend、p prior、SciPy 缓存语义和坐标导出与父后端兼容。components 打包为一次主机标量传输，梯度打包为一次传输。

`cuda_pair_objective.cu` 使用 `torch.utils.cpp_extension.load` 构建，需要 `CUDA_HOME=/usr/local/cuda`、已安装的 nvcc 和 ninja。`MAX_JOBS=2` 与 `TORCH_EXTENSIONS_DIR` 是进程级设置，并指向 `028/work/fused_build/` 下。编译使用 float64，不使用 `--use_fast_math`。

绑定检查每个 tensor 都与 x 位于同一设备，安装 `c10::cuda::CUDAGuard`，并在 `getCurrentCUDAStream` 上启动；它不假定默认 stream。full 与 sparse 调用之间除正常 stream 依赖外不插入同步，因此调用方保留显式同步的计时边界。

## 验证门

1. 小型夹具：归档 NumPy 的 value/components/gradient、有限差分、重合拷贝、零距离排斥、整条染色体交换以及 `p=0.5` 局部计数不变性必须通过。
2. 真实 020 的 1 Mb accepted-0240 检查点：使用 block/repulsion size 65,536 和一个 BLAS 线程与归档 CPU 比较。冻结 gate 为归一化计数误差 <=1e-10、梯度最大误差 <=1e-8，原始 component 误差受 1e-10 相对缩放容差限制。
3. 20 kb 审计：断言 `N=131700`、`E=8672379150`、全部四种拷贝组合、精确的原始/聚合/端点预算、正分组保护、稀疏 CSR 覆盖以及没有稠密 E 大小分配。
4. 只有通过这些 gate 后，才可考虑 20 kb objective 冒烟检查或正式优化器运行。240 次迭代 / 750 次函数调用预算保持不变。该设计本身不推出任何 walltime 或 20 kb 可行性结论。
