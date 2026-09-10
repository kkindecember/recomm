# 第三方参考代码

`RaSeRec/SOURCE.json`记录作者仓库、固定提交和下载归档哈希，原MIT许可证保留于`RaSeRec/LICENSE`。

Git仅保留来源记录、许可证、作者README、相关默认配置，以及`layers.py`、`duorec.py`、`raserec.py`三个参考源码文件。其余下载的RecBole框架保留在实验机器，通过`.gitignore`排除；这里不是可独立运行的完整作者工程。

本项目实际执行`experiment/phase18/core/raserec_author.py`中的提取实现及GRAM适配。`test_raserec_gram.py`读取保留的`layers.py`和`duorec.py`，用AST比较核对作者核心类的一致性。需要完整上游工程时，应按`SOURCE.json`中的URL和commit另行获取。
