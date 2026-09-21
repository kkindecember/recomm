# 第三方参考代码

`RaSeRec/SOURCE.json`记录作者仓库、固定提交和下载归档哈希，原MIT许可证保留于`RaSeRec/LICENSE`。

Git仅保留来源记录、许可证、作者README、相关默认配置，以及`layers.py`、`duorec.py`、`raserec.py`三个参考源码文件。其余下载的RecBole框架保留在实验机器，通过`.gitignore`排除；这里不是可独立运行的完整作者工程。

本项目实际执行`experiment/phase18/core/raserec_author.py`中的提取实现及GRAM适配。`test_raserec_gram.py`读取保留的`layers.py`和`duorec.py`，用AST比较核对作者核心类的一致性。需要完整上游工程时，应按`SOURCE.json`中的URL和commit另行获取。

## 第20阶段外部仓库

[upstream_sources.json](upstream_sources.json)记录LIME-Rec和SEAR的来源、固定提交及存在的许可证文件哈希。归档时两者的受版本管理文件均无本地修改。完整下载目录、内部`.git`、数据、模型和运行输出通过根目录`.gitignore`排除，不作为子模块或源码副本提交。

LIME-Rec由第20阶段脚本直接导入，需要恢复完整作者工程。SEAR用于固定版本源码核查和近邻方法对照准备，下载源码不代表已经完成其完整基线训练；固定版本仓库未发现根目录许可证文件。

在项目根目录、对应目录尚不存在时执行：

```bash
git clone https://github.com/Double-wk/LIME-Rec.git third_party/LIME-Rec
git -C third_party/LIME-Rec checkout --detach af348958bef4a580bc775edcd27fab91e71b529a
git clone https://github.com/guanwei49/SEAR.git third_party/SEAR
git -C third_party/SEAR checkout --detach 4dc184400d96a93d74d14270f5f8c869b667a835
```

已有目录时先核对`git -C <目录> status --short`和`rev-parse HEAD`，保留本地修改，不覆盖运行中的工程。按各自README准备依赖；训练数据、下载权重和本地实验缓存仍需另行准备。第20阶段配置中的绝对路径、GPU绑定和恢复检查点是实验机器记录，迁移机器时需要调整。
