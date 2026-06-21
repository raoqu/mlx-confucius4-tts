目标：本项目是从 https://github.com/netease-youdao/Confucius4-TTS （本机路径：~/research/Confucius4-TTS）基础之上，参考 https://github.com/antirez/ds4 的项目实现思路，将本项目（ Confucius4-TTS）改造成C++为Apple Silicon特别优化的可独立运行不依赖 python 环境的项目化。在尽可能不降低TTS模型精度的同时追求极致的性能，使得RTF最小。迁移时避免追求一次性完成，而是拆解成每一步可独立验证的步骤，目前已经生成了计划文档 docs/PLAN.md

原则：除了为Apple Silicon所做的性能优化，需要忠实于原项目的算法和逻辑实现，如果原项目依赖三方的库和模块，可以考虑下载对应库和模块的代码实现来解耦依赖。

工作流程：以后每次功能实现完成后，默认提交代码（git commit）。按功能拆分为独立的提交，提交信息使用 `c4tts: <简述>` 的格式；除非用户另有说明，直接提交到 main 分支。