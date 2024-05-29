## 如何发现好的问题

作者：[Zitao Shuai](https://zitao-shuai.github.io/) (ztshuai@umich.edu or zitao.shuai@zju.edu.cn)

我想以我在机器学习/数据科学领域的研究经验为例来总结这个问题。在这一节里面，我想首先从本科导师对我的启发最大的一句话出发，介绍从研究工作的 position 角度考虑问题的重要性，并比较我不同研究阶段的 taste，来总结什么是一个好问题。随后我会从一些案例出发，总结如何去发现一个还不错的问题。

### 关注研究问题所处的位置

我本科找导师聊如何读文章的时候，老师告诉我要关注问题的 position。这个对于大三的我来说无疑是非常有启发性：科研重要的不是 ”我认为这个问题没做过所以可以做”，而是**“大家觉得这个问题有意义所以值得做”**。

从这两个不同的思路出发，我们很容易得到两个完全不同的灌水路径。从“我认为这个问题没做过所以可以做” 出发，我们很自然就是想到 A 方法加上 B问题，甚至 A+C 方法来解决 B 问题。这样的坏处很显而易见，我们的工作就是 literally 在灌水，甚至都无法搞清楚 B 问题的核心挑战在哪里。为了让这种灌水文章能够中，我们需要：调参甚至优化 table 来让提出所谓的新方法打败 baseline；写作的时候随意 make claim，最后这些都无法被实验证明，连现有工作都很难被用来证明我们的 claim。例如，我把24年刚出的 MAMBA [2] 模型用到肠息肉图像分割领域，就是个经典 A+B 的应用。如果我是审稿人，我会问为什么 transformer 作为 backbone 就不行呢？MAMBA 有什么独有的优势会使得在这个问题上的效果很棒呢？如果遇到审稿人抓着 novelty 的点给我们硬塞 strong reject，那真的很难救回来。

作为反例，就是我自己和工友在大二的一系列关于科研的尝试。例如：“利用神经网络学习股票市场的高频交易数据中的趋势性时间序列信号”；例如 “利用神经网络学习数据库索引到数据的映射关系”；例如 “利用魔改的 xxx-tree 预测公司财务困境问题”； 例如“利用 A 数据结构压缩时空数据，并用 B 算法学习时空图的动态演化”。当然，如果文章写得好，这些工作都可以被写的很有 motivation，从而被认可。同样是用神经网络学交易趋势，修老师在金融最顶级的刊物 Journal of Finance 上的 [1] 的写法是：“交易员有丰富的交易经验，对数据走势这样的交易信号学的很好，因此我们可以从这个现象作为启发，用卷积神经网络来学习这样的信号”。

如果从“大家觉得这个问题有意义所以值得做”出发，我们的研究一开始就是问题-导向的，因此在最坏的情况下，我们也最多做出对问题的 “A+B”的灌水。从“有意义的问题出发”，我们至少会思考到如下的内容：

1. 我们的工作处于什么样的位置：是 unify 了现有的工作，不仅提升了性能，还在方法上提出了更简洁、统一的形式吗？是发现了我们解决的问题里面，过去所没有发现和探究过的挑战吗？是提出了一个新的实际的任务，对我们研究的领域做了补充和完善吗？或者有大佬力挺我们，投了个大刊挖了一个新坑吗？
2. 我们的工作的影响力会体现在哪里，我们的 problem insight 很有趣吗？ 我们的工作容易在 gscholar 上被别人搜到吗？我们的方法有启发吗？ 我们这个工作有让更多人来关注我们的问题吗？ 我们的表现足够低从而会常被别人当作 baseline 来比吗（当然需要比现有 baseline 高）？这些主要是和引用有关，如果都靠兄弟姐妹来引，估计也很难破百。 
3. 领域核心圈认不认可我们的工作，他们对我们研究的问题的看法是怎么样的。例如，我作为大头兵，看了一篇 cv 的工作，认为可以试试完全用 synthetic data pre-train 一个领域大模型。我的导师作为 rising star，认为大家关注的问题是 synthetic 数据的 reliability，所以认为我们可以拆分这个工作。挖坑的大佬会说，reliability 的保证和评估本身也是困难的，如何定义并解决这个问题是非常关键的。




[1] Jiang J, Kelly B, Xiu D. (Re‐) Imag (in) ing Price Trends[J]. The Journal of Finance, 2023, 78(6): 3193-3249.

[2] Gu A, Dao T. Mamba: Linear-time sequence modeling with selective state spaces[J]. arXiv preprint arXiv:2312.00752, 2023.

[3] Bannur S, Hyland S, Liu Q, et al. Learning to exploit temporal structure for biomedical vision-language processing[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 15016-15027.
