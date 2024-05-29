## 如何设计实验

在泛人工智能领域，有时候并不太需要会设计实验，在刷点的任务中，我们往往选择 follow 一篇文章的实验流程，并让自己的模型的在各个评估 metric 下的点数更高即可。然而，如果我们并不是去 follow 别人的工作，而是提出一个新设定呢？又或者说，我们突然发现一个很有趣的问题值得研究，需要挖坑，以至于需要提出新的评估方式呢？或者我们的跨学科合作者有新的数据集，准备一起发新的 benchmark 赚引用呢？因此，和自然科学学科类似，AI 领域实际上也需要严谨的实验设计。在这篇博客中，我主要总结这两年来的经验，有关实验的作用，设计实验的方法，以及如何让表格看起来没问题。

### 实验服务于 claim 

我们回顾一下本科的第一篇文章，在 introduction 里面是不是经常会有诸如 “现有的方法不能处理这个流式数据”， “困难样本会极大阻碍模型学习可泛化的表征从而影响预训练结果”， 或者 “我们的方法通过解决 A 来提高了 xxx 能力”。这些都是我们所做的 claim （或者说“声称”/"声称的结论"/“预先声称的结论”），是需要被实验或者理论证明的。在 NIPS 的投稿中，我们被要求填写一个 checklist，这个 checklist 里面有问我们是否对每一个 claim， 都有相应的证据来证明。由此可见回应 claim 的重要。

那么我们如何去证明我们做的 claim 是正确的呢？首先大多数人 claim 很难用 theory 去证明的，这主要是在非理论的文章中，我们并没有足够的篇幅去完整定义、刻画一个问题并且提出 theorem 等等并证明。比方说“困难样本会极大阻碍模型学习可泛化的表征从而影响预训练结果”，那我们首先要开一个 preliminary section，定义什么是困难样本，什么是可泛化，再定义我们的目标等等。

事实上，最后我们通常会也只能选择用实验来证明 claim。所以我们设计实验的主线不是仅仅看 state of the art (SOTA) 有多高，而是看我们的表格有没有充分证明我们的 claim，从而支持我们的文章的贡献。紧跟这个主线，我们设计的实验往往能很好的撑住 introduction 的故事线。

### 如何设计好的实验

那么我们应该如何设计实验呢？我们从一个领域由蓝海到红海到大家转行挖坑为线来开始梳理：

大佬提出一个问题并给出最基础的 baseline 之后 （这个大概相当于 out-of-distribution learning/optimization 里面的 ERM [1], 联邦学习里面的 FedAvg, 金融时间序列里面的 GARCH），我们要做的就是把点给刷高。这个时候我们的套路就是先刷点再写故事，基本思路是：（a). main table， 各个指标全部 beat; (b). ablation study，证明我们提的模块全部有用。相应的，introduction 里面我们的写法主要是“为了提高表现，所以加入 xx 模块来 xxx ”。这个时候由于做的人不多，我们往往用的是一些很显然的做法，只需要消融实验证明这个做法有效即可，不需要明确指出这个做法解决的是什么问题。例如：在联邦无监督对比学习的时候，单一 client 的样本多样性不足，所以我们引入其他 client 的表征，既可以一定程度保证安全性，也可以增加样本多样性 [2]。这个做法是很容易想到的，所以我们不需要花很大力气在“样本多样性”上面。

在一个问题变得很多人研究之后，往往 SOTA 会虚高 （可能存在有些人优化表格过于黑的情况），各种常规角度的初步尝试也有人做过。那么破局的办法就是标新立异，提新的角度来解决这个问题。往往这个时候我们的点不需要非常高，但是我们提的角度/特性/问题/挑战一定要有趣，更重要的是得到证明。例如 [3] 提到了 flatness 在泛化任务上的提升作用，但是这个和泛化本身并没有太直接的关系。因此这篇文章除了 main table 来说明方法的有效性、ablation study 说明提出的模块的有效性之外，还一直围绕 flatness 点设计了很多实验。

在一个任务被卷得白热化之后，我们往往会提出新设定，这设定一方面需要有意义，一方面需要未解决。例如我大四有一篇一直被拒只能扔掉的文章，考虑的是“泛化模型在流式数据下持续学习可泛化的知识”。这其实是问题层面的 A+B 的工作。复盘一下，如果我现在重新做这个问题，我首先要考虑：泛化模型是否一定会在持续学习场景失效？持续学习方法是否真的不能直接被迁移过来？所以我第一个表格的标题将会是：泛化模型在持续学习场景有灾难性遗忘问题；第二个表格的标题会是：现有持续学习方法不能直接迁移过来；第三个表格更重要，他是为了指出是什么问题导致持续学习的方法不能迁移过来：（例如）持续学习关注的是记住当前数据集，而不是提取可泛化的知识。这样，通过在 main table 之前加入这些 empirical 分析，我们后面的论证会更有力。

如果在新设定也层出不穷之后，往往一个领域就差不多该被做死了。这个时候可能有大佬带着数据集出来挖新坑，在我熟悉的领域，比较好的坑有：[4] [5] [6]。在 medical AI 里面，挖坑的工作往往需要大佬+数据集，technique contribution 相对更不重要一些。实验上，主要是在提的数据集上跑各种实验，验证一些结论。这种工作作为大头兵，不一定有机会能lead，但是能蹭的话就很好。



### 如何让表格看起来没问题

表格的问题主要是出现在提新设定/问题的时候，因为没有过去工作的 baseline 的数据可以直接抄。自己花很多计算资源去将相关工作迁移过来并复现，只是其中一个麻烦的地方。事实上，不少工作直接迁移过来，点数会非常难看。为了防止审稿人认为 baseline 写的有问题，我们一方面要保证代码没问题，一方面可以放一个 lower bound 放一个 upper bound, 然后再给 baseline 搜一下参数，给他尽量提起来。

除了很奇怪的点数之外，baseline 最好在 general task 呈现出：lower bound < weak baseline < strong baseline < ours; 在特定目标，例如B，C方法优化了目标A，A是我们认为提升模型性能的关键或者是另一个评估指标。所以我们希望在A上的看到如下的表现: lower bound < strong baseline < B < C < ours，同时在 general task 上面我们也希望看的 trade-off: lower bound < B < C < strong baseline < ours.

如果我们的数据实在太奇怪，要不是代码有问题，要不就是这个指标不能报（例如 TPR 太低的时候用 F1 就好看很多），或者之前工作的黑心哥们儿做的太不地道了。最好的方式是尽量让评估没问题，然后报一个大家都 acceptable 的结果。





[1] Vapnik V. Principles of risk minimization for learning theory[J]. Advances in neural information processing systems, 1991, 4.

[2] Zhang F, Kuang K, Chen L, et al. Federated unsupervised representation learning[J]. Frontiers of Information Technology & Electronic Engineering, 2023, 24(8): 1181-1193.

[3] Cha J, Chun S, Lee K, et al. Swad: Domain generalization by seeking flat minima[J]. Advances in Neural Information Processing Systems, 2021, 34: 22405-22418.

[4] Feuerriegel S, Frauen D, Melnychuk V, et al. Causal machine learning for predicting treatment outcomes[J]. Nature Medicine, 2024, 30(4): 958-968.

[5] Ktena I, Wiles O, Albuquerque I, et al. Generative models improve fairness of medical classifiers under distribution shifts[J]. Nature Medicine, 2024: 1-8.

[6] Komorowski M, Celi L A, Badawi O, et al. The artificial intelligence clinician learns optimal treatment strategies for sepsis in intensive care[J]. Nature medicine, 2018, 24(11): 1716-1720.
