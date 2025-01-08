@[TOC](文章目录)

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 前言
请先思考几个问题：

* 你是否在全网苦寻【图像去噪（Image Denoising）】的相关资料？
* 你的目标是否是看懂【图像去噪（Image Denoising）】的相关论文，复现代码，跑出结果，并试图创新？
* 你是否需要发表【图像去噪（Image Denoising）】的相关论文毕业？
* 你是否需要做【图像去噪（Image Denoising）】的相关项目，开发软件，研究算法，获得专利或者软著？
* ......

只要是与【图像去噪（Image Denoising）】有关的问题，那么请继续往下看。

以刚读研的研究生为例，进组后导师让你研究与【图像去噪（Image Denoising）】的相关课题，但又没有时间指导你，只能自己研究。

这个时候你该怎么办呢？刚入门的小白，什么也不了解，只能是自己搜集资料或者问师兄师姐。

如果你问师兄师姐，怎么搞科研啊？如何看懂论文啊？如何读懂复杂的代码啊？

他们大概率会回复你：硬啃。一句一句地读论文呢，一行一行地理解代码。无他，唯手熟尔。

你摇了摇头，觉得他们“自私”，没有把“看家本领”传授于你。于是，你开始像无头苍蝇一样，试图寻找好的学习方法，希望通过一种好的学习方法来提升科研效率。

想法很好，但浪费时间和精力，而且没有开始真正的学习，只是在学习的边缘徘徊。其实，就差一步，管它科研难不难，学就完事了。

学习无捷径，大道至简，先学起来，再慢慢修正学习路线，最后到达顶峰。

很幸运你看到了本文，本专栏的意义就在于用我走过的路，带领你开始学习，避免“找方法找资料”这种费时费力的弯路。领先实验室同门，早发论文，早做项目，受导师青睐，尽早达到毕业要求好去实习。

三年时间转瞬即逝，必须马上开始！而你要做的，只是认认真真的读完每一篇专栏内的文章即可，相信你一定会有收获，科研能力越来越强，顺利完成你的目标。

说点肺腑之言，感兴趣的话继续往下看。
# 适配人群

急需入门【图像去噪（Image Denoising）】的朋友，具体为：

* 看不懂论文、写不出代码、无创新思路的新手小白，希望快速入门
* 课题与去噪相关，需要发论文毕业的本科生和研究生（硕士、博士），缺乏科研能力以及论文写作经验的实验室难兄难弟
* 导师放养，不知道选择哪个方向入手的迷茫者
* 需要论文、专利、软著等评职称的相关人员
* 有做去噪相关项目的本科生、研究生（硕士、博士）、企业开发人员、算法工程师等

如果你符合上面的某一条，不用担心，继续往下看即可。
# 专栏简介
专栏名称暂定为【Pytorch深度学习图像去噪算法100例】。顾名思义，共三点限制：
1. 专栏内涉及的算法都是**基于深度学习**的图像去噪算法。非深度学习的可能只介绍BM3D，用做实验对比方法的baseline（类似超分的Bicubic）。
2. 所有算法都是**基于Pytorch框架**实现的。论文公开的源码是Pytorch的我们就基于源码复现，不是用Pytorch框架实现或者未公开源码的算法，我会用Pytorch重新复现一遍，以达成代码的统一，方便大家做实验。
3. 图像去噪领域**顶刊顶会文章精选100篇论文，复现100个源码**（目前更新中，等达成后再继续收录）。

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

专栏内文章主要为两部分：【论文精读】与【论文复现】

* 论文精读：读懂论文，总结提炼，聚焦核心内容，不只是全文翻译

* 论文复现：跑通流程，源码解析，提升代码能力，得到去噪结果以及指标计算

综合而言，**从大到小拆解模型结构，从小到大实现模型搭建**。实现论文与源码的统一，深入理解论文行文逻辑与代码实现逻辑，融汇贯通二者思想，并学以致用。

更具体详细的内容见“阅读方法”。
# 专栏亮点
 * <font color = 'red'>**省时：图像去噪领域论文全覆盖，算法模型全，主流的算法模型都会涉及结合能搜集到的资料深入浅出解读，按顺序阅读即可，节省搜论文的时间。【论文复现】文章末尾会提供训练好各放大倍数下的最优性能（PSNR/SSIM）模型权重文件，读者可以下载直接使用，方便论文中实验部分方法对比（视觉效果和量化指标）或直接应用模型超分自己的图像数据。**</font>
* <font color = 'red'>**省力：大白话讲解，手把手教学，带你读懂论文，跑通代码，提升代码能力，理解文章的算法和创新点，避免一个人对着论文死磕但还是不理解的情况出现。按部就班，肯定能跑出好的模型结果，帮助你了解论文内容和结构，积少成多，学会写论文。同时，让你少走弯路，避免模型调参四处碰壁，涉及的模型都提供训练好的权重文件，可以直接拿来测试自己的图像。**</font>
*  <font color = 'red'>**省事：努力做全网最全最好最详细的【图像去噪】专栏，专栏永久更新，第一时间更新最新的图像去噪领域，关注订阅后一劳永逸，关注账号后更新了会私信告知。有问题可以随时留言交流。**</font>
# 阅读方法
**【论文精读】** 文章结构按照论文的结构包含**全文翻译、关键信息提炼，信息总结等**。作者自己的总结以及注解以<font color = 'red'>**红色加粗**</font>和<font color = 'green'>**绿色加粗**</font>呈现。**阅读和之前工作相比创新改进的地方，如网络结构（重点）、损失函数等。**

根据自身情况，选择以下 **【论文精读】**  的阅读方式：

1. 有写论文需求的朋友：对照原论文，全文阅读。除了看懂论文提出的方法外，还需要培养“讲故事”能力，深化论文细节，为论文写作做准备。
2. 想快速理解文章的朋友：略读文章翻译部分，重点看<font color = 'red'>**加粗红字**</font>的提炼总结。
3. 只想了解文章创新：看摘要和介绍部分末尾提出的创新贡献+网络结构，然后直接跳转到文末的复现文章。

> 上述三种情况对应的论文结构：
> 1. Abstract （1、2、3；无论哪种情况都应该看摘要）
> 2. Introduction（1、2；搞懂motivation，找创新思路）
> 3. Related Work（1）
> 4. The Proposed Method（1、2、3；1全看，2和3看重点）
> 5. Experimental（1；实验部分重点在对应的【论文复现】文章中）
> 6. Conclusion （1）
> 补充：如果你符合第一种情况，那么应该以审稿人的角度来看文章。如果你是审稿人，你会先看哪里，后看哪里，重点看哪里，以及给这篇文章什么审稿结果？问题的答案就是你在写论文的时候应该注意的东西。

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

**【论文复现】** 文章结构：
1. 跑通代码：拿到任何一个代码项目，首先要做的就是根据README.md先跑通，【复现】文章的第一部分就相当于README，即使用手册。每篇复现文章的第一部分均是详细的跑通步骤，包括**设置配置文件、设置数据集和预训练模型路径、需要的依赖库、报错解决**等。根据路径放置好数据集，跑通训练代码，训练完跑通测试代码。如果有已经训练好的模型，可以直接推理测试。
2. 代码解析：详细讲解全部代码，注释丰富，网络结构以及相关参数与论文一致。代码结构一般为**数据预处理、网络结构实现、训练、测试、其他补充代码**。重点在于**网络结构以及特殊损失函数的实现**，其他部分大同小异。尽量复现出与原论文一致的结果。
3. 总结与思考：记录复现过程中遇到的问题，思考可能的改进提升。附带完整代码和训练好的模型权重文件下载链接。

根据自身情况，选择以下 **【论文复现】**  的阅读方式：

1. 只需要结果（去噪后的结果图像、计算测试集指标PSNR/SSIM）：只看1. 跑通代码
2. 为了提升代码能力，实现论文与代码对应，改进结构：1.2.3全看
# 定价理由
专栏原价~~299.9~~，目前特价199.9，订阅满50人恢复原价，先到先得！！！

为什么选择本专栏？

* 和其他能搜到的去噪专栏相比，本专栏文章**数量更多，质量更高**。预计更新100个图像去噪算法，每个算法两篇文章，**平均每篇文章的价格为199.9/200 ≈ 0.99元**，文章保质保量，专栏文章平均质量分为96。少吃半顿海底捞，就能解决一个大麻烦。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cf094036cb2b4bbaa26591554844a769.png)
排行榜前列：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ce1b16885adf4c97842bbe8a1254e224.jpeg)

其他兄弟专栏：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bc952556b61c42418cd7181335eddfba.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/afe9e6359a224c00860157a74058e568.jpeg)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fed8d0e09e264eed8192dd311224605c.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ce0f6bf24c174da98a48d7e12383c584.png)

* 以硕士期间从研一到研二下学期一年半180天发出小论文计算，平均每天为199.9/180 = 1.11。也就是说，**每天只花1.11元就可以获得一个一劳永逸的学习专栏**，何乐而不为？对发论文、研究算法、做项目、以后找工作都有帮助，很快就可以回本。
* 以时间计算，180天只需每天花一小部分时间复现一篇论文，努力一下可以复现两篇，何况也不一定专栏内所有文章都看。试想一下，当其他人还在花费大量时间啃论文啃代码焦头烂额时，你已经不费吹灰之力，领先于人了。
* 为知识付费，为学习投资，**形成知识体系，提升代码能力**，还可以拓展人脉（订阅可加交流群），潜在价值远非价格可比。
* 避免毕业困难，自信受挫，失去学习热情
# 品质承诺
1. 【论文精读】和【论文复现】文章**保质保量**。文章质量参考两篇DnCNN的置顶文章，**请先试读**：
[【图像去噪】论文精读：Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising（DnCNN）](https://blog.csdn.net/qq_36584673/article/details/139575260)
[【图像去噪】论文复现：新手入门必看！DnCNN的Pytorch源码训练测试全流程解析！为源码做详细注释！补充DnCNN-B和DnCNN-3的模型训练和测试！附各种情况下训练好的模型权重文件！](https://blog.csdn.net/qq_36584673/article/details/139743314)
2. 订阅后永久**免费阅读专栏内全部文章**，**免费获取专栏内全部代码**。
3. 专栏内文章如不符合上述试读文章的质量，**可全额退款**，**无任何风险。其他情况不予退款**。
# 关于更新
本专栏**永久更新**。为了保证文章质量，更新速度会慢一些。更新了会群发消息通知，所以请订阅专栏的朋友一定要关注我，否则收不到更新通知。

# 关于答疑
宗旨：**先仔细阅读文章，有问题再问**。

基础薄弱的同学先学深度学习基础，搞明白框架和执行逻辑，重点为pytorch（数据处理DataLoader、模型实现nn、训练相关）、cv2和PIL等图像处理基础，有的问题搜一搜，查一查，把基础打好。

欢迎提出以下问题：
* 专栏内文章中出现的错误和表意不明的内容；
* 代码报错；
* 与算法模型相关的讨论，改进等；
* 希望我后续出的文章，方向，功能需求等；
* 友好的讨论，各种方面都可以，论文、投稿、读研读博、当老师等；

<font color = 'red'>**符合以上条件的，我知无不言，作为一名高校教师，我很愿意和友好的学生沟通！工作很忙，空闲时间看见了就会回复。**</font>

以下评论和问题不回：
* **不友好的攻击与谩骂，没有礼貌的**；比如：“这么简单的东西还收费”。你觉得简单，还有很多需要入门的小伙伴不觉得简单，不要用你的标准来评判他人；
* **不看文章一通乱问的**；答案就在文章中，仔细阅读，文章写的很细致，问我我给你回答没有你仔细阅读效率高；
* **表述不明，完全不知所云的**；不明白你的问题，自然无法解答；
* **上来就要代码的，想白嫖的，套话的**；不要问我这个方案可不可以，能不能给个创新idea等厚脸皮的话。一切都要自己动手做，去试了才知道，谁有好的想法不都自己发文章了吗，还能告诉你吗？
* **我也不会的**；我不是神仙什么都会，可以讨论，但不能要求我一定能回答上；

<font color = 'red'>**订阅专栏后的答疑是作为附加的增值服务，我没有义务什么离谱的问题都回答你，即使你花了钱也不行，每个人的标准不同，求同存异，友好相处，有问题有意见可以提，但别找骂。**</font>
# 环境配置
项目环境：

* 编译器：pycharm
* cuda：torch 1.12.1
* 操作系统：windows11本地运行（RTX 4070Ti Super）本地运行或Linux服务器（4个Titan RTX GPU）

只要不是太差的显卡，专栏内的算法都可以跑。

不建议CPU运行。

复现文章中提供的**代码在Windows和Linux下均可运行**！
# 去噪概述
噪声主要分为两类：
* 合成噪声（Synthetic Noise），一般指在“干净”图像上添加高斯白噪声（Additive White Gaussian Noise，AWGN）

* 真实噪声（Real Noise），即相机拍照时图像上的真实噪声。一般以真实图像作为带噪图像输入，mean作为Ground-truth。

对应噪声分类，产生了两个不同的研究方向以及算法：
* 基于合成噪声：一般是高斯白噪，便于量化和设计实验；如DnCNN，FFDNet

**合成噪声去噪示例**：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7d680b0fc8df4e4f94d61a12f9bc6bee.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1209b973c61c44fa8b10983bdc85a4bb.png)

* 基于真实噪声：真实世界的盲去噪问题，真实图像的噪声分布是未知的，可能是某些不同噪声类型和分布的叠加。有用泊松-高斯模拟，用ISP相机产生噪声的原理模拟等，属于摸着石头过河，需要很强的统计学、图像信号功底；如CBDNet，RIDNet，VDNnet

**真实噪声去噪示例**：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0723eec2d78e49288b438d3b16d1c5de.jpeg)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/79a9c078b05b488f96b94edd33ae52d7.jpeg)
SIDD数据集上SOTA方法视觉效果大对比：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3e8992140d4d4dcf9c2f6b4cea83baa1.jpeg)

* 以及两者通用的算法。
# 文章目录
图像去噪任务大体分为监督和非监督两类，监督类模型数据集中包含Ground-Truth，非监督一般没有Ground-Truth。

文章目录按年份和模型性能综合排序，按顺序阅读即可。
## 有监督
1. [（**TIP 2017**）【图像去噪】论文精读：Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising（**DnCNN**）](https://blog.csdn.net/qq_36584673/article/details/139575260)
2. [（**TIP 2017**）【图像去噪】论文复现：新手入门必看！**DnCNN**的Pytorch源码训练测试全流程解析！为源码做详细注释！补充DnCNN-B和DnCNN-3的模型训练和测试！附各种情况下训练好的模型权重文件！](https://blog.csdn.net/qq_36584673/article/details/139743314)
3. [（**TIP 2018**）【图像去噪】论文精读：**FFDNet**: Toward a Fast and Flexible Solution for CNN based Image Denoising](https://blog.csdn.net/qq_36584673/article/details/139766955)
4. [（**TIP 2018**）【图像去噪】论文复现：适合新手小白的Pytorch版本**FFDNet**复现！详解FFDNet源码！数据处理、模型训练和验证、推理测试全流程讲解！新手小白都能看懂，学习阅读毫无压力，去噪入门必看！](https://blog.csdn.net/qq_36584673/article/details/139789656)
5. [（**CAAI Transactions on Intelligence Technology 2019**）【图像去噪】论文精读：Enhanced CNN for image denoising（**ECNDNet**）](https://blog.csdn.net/qq_36584673/article/details/142256089)
6. [（**CAAI Transactions on Intelligence Technology 2019**）【图像去噪】论文复现：Pytorch实现**ECNDNet**！入门项目，适合新手小白学习使用，Windows和Linux下均可运行！附训练好的模型文件，可直接得到去噪结果和指标计算！](https://blog.csdn.net/qq_36584673/article/details/142257521)
7. [（**CVPR 2019**）【图像去噪】论文精读：Toward Convolutional Blind Denoising of Real Photographs（**CBDNet**）](https://blog.csdn.net/qq_36584673/article/details/141162868)
8. [（**CVPR 2019**）【图像去噪】论文复现：适合新手小白的Pytorch版本**CBDNet**复现！轻松跑通训练和测试代码！简单修改路径即可训练自己的数据集！代码详细注释！数据处理、模型训练和验证、推理测试全流程讲解！](https://blog.csdn.net/qq_36584673/article/details/141191222)
9. [（**ICCV 2019**）【图像去噪】论文精读：Real Image Denoising with Feature Attention（**RIDNet**）](https://blog.csdn.net/qq_36584673/article/details/141261836)
10. [（**ICCV 2019**）【图像去噪】论文复现：适合新手小白的Pytorch版本**RIDNet**复现！轻松跑通训练和测试代码！RIDNet网络结构实现拆解！简单修改路径即可训练自己的数据集！模型训练推理测试全流程讲解！](https://blog.csdn.net/qq_36584673/article/details/141284977)
11. [（**NeurIPS 2016**）【图像去噪】论文精读：Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections（**REDNet**）](https://blog.csdn.net/qq_36584673/article/details/141322304)
12. [（**NeurIPS 2016**）【图像去噪】论文复现：Pytorch实现**REDNet**的三种结构！简单修改路径即可跑通全部代码并训练自己的数据集！支持灰度图和RGB图训练！附训练好的模型文件可直接测试图像得到去噪结果以及评价指标！](https://blog.csdn.net/qq_36584673/article/details/141471808) 
13. [（**ICCV 2017**）【图像去噪】论文精读：**MemNet**: A Persistent Memory Network for Image Restoration](https://blog.csdn.net/qq_36584673/article/details/141397975)
14. [（**ICCV 2017**）【图像去噪】论文复现：全网最细的Pytorch版本实现**MemNet**！论文中的网络结构图与代码中的每个变量一一对应！实现思路一目了然！附完整代码和训练好的模型权重文件！](https://blog.csdn.net/qq_36584673/article/details/141423575)
15. [（**CVPR 2018**）【图像去噪】论文精读：**xUnit**: Learning a Spatial Activation Function for Efficient Image Restoration](https://blog.csdn.net/qq_36584673/article/details/141557145)
16. [（**CVPR 2018**）【图像去噪】论文复现：**代替ReLU**！Pytorch实现即插即用激活函数模块**xUnit**，并插入到DnCNN中实现xDnCNN！](https://blog.csdn.net/qq_36584673/article/details/141638907)
17. [（**CVPR 2018**）【图像去噪】论文精读：Multi-level Wavelet-CNN for Image Restoration（**MWCNN**）](https://blog.csdn.net/qq_36584673/article/details/141592164)
18. [（**CVPR 2018**）【图像去噪】论文复现：小波变换替代上下采样！Pytorch实现**MWCNN**，数据处理、模型训练和验证、推理测试全流程讲解，无论是科研还是应用，新手小白都能看懂，学习阅读毫无压力，去噪入门必看！](https://blog.csdn.net/qq_36584673/article/details/141600616)
19. [（**CVPR 2017**）【图像去噪】论文精读：Learning Deep CNN Denoiser Prior for Image Restoration（**IRCNN**）](https://blog.csdn.net/qq_36584673/article/details/141592164)
20. [（**CVPR 2017**）【图像去噪】论文复现：支持任意大小的图像输入！四十多行实现Pytorch极简版本的**IRCNN**，各种参数和测试集平均PSNR结果与论文一致！](https://blog.csdn.net/qq_36584673/article/details/141672251)
21. [（**ECCV 2020**）【图像去噪】论文精读：Spatial-Adaptive Network for Single Image Denoising（**SADNet**）](https://blog.csdn.net/qq_36584673/article/details/141676987)
22. [（**ECCV 2020**）【图像去噪】论文复现：三万字长文详解**SADNet**的Pytorch源码！全网最详细保姆级傻瓜式教程，新手小白也能看懂，代码逐行注释，跑通代码得到去噪结果毫无压力！网络结构图与模型定义的量一一对应！](https://blog.csdn.net/qq_36584673/article/details/141701661)
23. [（**ICLR 2019**）【图像去噪】论文精读：Residual Non-local Attention Networks for Image Restoration（**RNAN**）](https://blog.csdn.net/qq_36584673/article/details/141679352)
24. [（**ICLR 2019**）【图像去噪】论文复现：非局部注意力机制提升去噪性能！Pytorch实现**RNAN**，解决out of memory问题，论文中结构图与代码变量一一对应，清晰明了保证看懂！附训练好的模型文件！](https://blog.csdn.net/qq_36584673/article/details/141821026)
25. [（**CVPR 2020**）【图像去噪】论文精读：Transfer Learning from Synthetic to Real-Noise Denoising with Adaptive Instance  Normalization（**AINDNet**）](https://blog.csdn.net/qq_36584673/article/details/143700190)
26. [（**NN 2020**）【图像去噪】论文精读：Attention-guided CNN for Image Denoising（**ADNet**）](https://blog.csdn.net/qq_36584673/article/details/142288366)
27. [（**NN 2020**）【图像去噪】论文复现：注意力机制助力图像去噪！**ADNet**的Pytorch源码复现，跑通全流程源码，补充源码中未提供的保存去噪结果图像代码，ADNet网络结构图与源码对应，新手友好，单卡可跑！](https://blog.csdn.net/qq_36584673/article/details/144584512)
28. [（**NeurIPS 2019**）【图像去噪】论文精读：Variational Denoising Network: Toward Blind Noise Modeling and Removal（**VDNet**）](https://blog.csdn.net/qq_36584673/article/details/141321906)
29. [（**NeurIPS 2019**）【图像去噪】论文复现：包能看懂！**VDNet**的Pytorch源码全解析！逐行详细注释，理论与代码结合，提升代码能力！](https://blog.csdn.net/qq_36584673/article/details/142100254)
30. [（**TPAMI 2021**）【图像去噪】论文精读：Plug-and-Play Image Restoration with Deep Denoiser Prior（**DRUNet**）](https://blog.csdn.net/qq_36584673/article/details/143836176)
31. [（**TPAMI 2021**）【图像去噪】论文复现：无偏置在超大噪声下也能有效去噪！**DRUNet**的Pytorch源码复现，跑通DRUNet源码，得到去噪结果和评价指标，可作为实验中的对比方法，源码结构梳理，注释清晰，单卡可运行！](https://blog.csdn.net/qq_36584673/article/details/143843511)
32. [（**CVPR 2021**）【图像去噪】论文精读：**SwinIR**: Image Restoration Using Swin Transformer](https://blog.csdn.net/qq_36584673/article/details/144179782)
33. [（**CVPR 2021**）【图像去噪】论文复现：Swin Transfomer用于图像恢复！**SwinIR**的Pytorch源码复现，跑通源码，测试高斯去噪，使用预训练模型得到PSNR/SSIM以及去噪后图像！](https://blog.csdn.net/qq_36584673/article/details/144181326)
34. [（**Machine Intelligence Research 2023**）【图像去噪】论文精读：Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis（**SCUNet**）](https://blog.csdn.net/qq_36584673/article/details/143771545)
35. [（**Machine Intelligence Research 2023**）【图像去噪】论文复现：Swin Transformer块助力图像去噪！**SCUNet**的Pytorch源码复现，跑通SCUNet源码，理论结构梳理，获得SCUNet去噪结果，可作为实验中的对比方法！](https://blog.csdn.net/qq_36584673/article/details/143788643)
36. [（**ECCV 2020**）【图像去噪】论文精读：Dual Adversarial Network: Toward Real-world Noise Removal and Noise Generation（**DANet**）](https://blog.csdn.net/qq_36584673/article/details/142812175)
37. [（**ECCV 2020**）【图像去噪】论文复现：双对抗网络去除和生成逼真噪声！**DANet**的Pytorch源码复现，用于去噪和生成逼真噪声，训练、测试、损失函数、整体架构逻辑详解，图文结合，源码注释详细，单卡可跑！](https://blog.csdn.net/qq_36584673/article/details/143501371)
38. [（**ECCV 2020**）【图像去噪】论文精读：Learning Enriched Features for Real Image Restoration and Enhancement（**MIRNet**）](https://blog.csdn.net/qq_36584673/article/details/142151010)
39. [（**ECCV 2020**）【图像去噪】论文复现：全网最细！**MIRNet**的Pytorch源码复现全记录！论文中模型结构图与代码变量一一对应，保证看懂！踩坑报错复盘，一一排雷，Windows下也能轻松运行，代码逐行注释！](https://blog.csdn.net/qq_36584673/article/details/142176803)
40. [（**CVPR 2021**）【图像去噪】论文精读：Multi-Stage Progressive Image Restoration（**MPRNet**）](https://blog.csdn.net/qq_36584673/article/details/142499591)
41. [（**CVPR 2021**）【图像去噪】论文复现：三阶段网络！**MPRNet**的Pytorch源码复现，跑通全流程，图文结合手把手教学，由大到小拆解网络结构，由小到大实现结构组合，结构示意图与代码变量一一对应，全部源码逐行注释！](https://blog.csdn.net/qq_36584673/article/details/142935182)
42. [（**CVPR 2021**）【图像去噪】论文精读：**NBNet**: Noise Basis Learning for Image Denoising with Subspace Projection](https://blog.csdn.net/qq_36584673/article/details/141676949)
43. [（**CVPR 2021**）【图像去噪】论文复现：子空间投影提升去噪效果！**NBNet**的Pytorch源码复现，跑通代码，源码详解，注释详细，图文结合，补充源码中缺少的“保存去噪后的结果图像”代码，指出源码中模型实现的潜在问题](https://blog.csdn.net/qq_36584673/article/details/143462828)
44. [（**EUSIPCO 2022**）【图像去噪】论文精读：SELECTIVE RESIDUAL M-NET FOR REAL IMAGE DENOISING（**SRMNet**）](https://blog.csdn.net/qq_36584673/article/details/142812471)
45. [（**EUSIPCO 2022**）【图像去噪】论文复现：新手友好！**SRMNet**的Pytorch源码复现，跑通全流程源码，详细步骤，图文说明，注释详细，模型拆解示意图与代码一一对应，包能看懂，Windows和Linux下均可运行！](https://blog.csdn.net/qq_36584673/article/details/143182098)
46. [（**CVPR 2021**）【图像去噪】论文精读：Pre-Trained Image Processing Transformer（**IPT**）](https://blog.csdn.net/qq_36584673/article/details/143836138)
47. [（**CVPR 2021**）【图像去噪】论文复现：底层视觉通用预训练Transformer！**IPT**的Pytorch源码复现， 跑通IPT源码，获得测试集上平均PSNR和去噪结果，可作为实验中的对比方法！](https://blog.csdn.net/qq_36584673/article/details/143872450)
48. [（**ISCAS 2022**）【图像去噪】论文精读：**SUNet**: Swin Transformer UNet for Image Denoising](https://blog.csdn.net/qq_36584673/article/details/143211095)
49. [（**ISCAS 2022**）【图像去噪】论文复现：Swin Transformer用于图像去噪！**SUNet**的Pytorch源码复现，训练，测试高斯噪声图像流程详解，计算PSNR/SSIM，Windows和Linux下单卡可跑！](https://blog.csdn.net/qq_36584673/article/details/143231456)
50. [（**CVPR 2020 Oral**）【图像去噪】论文精读：**CycleISP**: Real Image Restoration via Improved Data Synthesis](https://blog.csdn.net/qq_36584673/article/details/142611564)
51. [（**CVPR 2020 Oral**）【图像去噪】论文复现：制作真实噪声图像数据集！**CycleISP**的Pytorch源码复现，跑通流程，可以用于制作自己的真实噪声图像数据集，由大到小拆解CycleISP网络结构，由小到大实现结构组合！](https://blog.csdn.net/qq_36584673/article/details/143140036)
52. [（**CVPR 2021 Oral**）【图像去噪】论文精读：Adaptive Consistency Prior based Deep Network for Image Denoising（**DeamNet**）](https://blog.csdn.net/qq_36584673/article/details/144328272)
53. [（**CVPR 2021 Oral**）【图像去噪】论文复现：扎实理论提升去噪模型的可解释性！**DeamNet**的Pytorch源码复现，跑通全流程源码，模型结构图与源码对应拆解，补充源码中没有的保存图像代码！](https://blog.csdn.net/qq_36584673/article/details/144349841)
54. [（**AAAI 2020**）【图像去噪】论文精读：When AWGN-based Denoiser Meets Real Noises（**PD-Denoising**）](https://blog.csdn.net/qq_36584673/article/details/144212410)
55. [（**AAAI 2020**）【图像去噪】论文复现：用Pixel-shuffle构建高斯噪声与真实噪声之间的联系！**PD-Denoising**源码复现，跑通训练和测试代码，可得到去噪结果和PSNR，理论公式与源码对应，图文结合详解！](https://blog.csdn.net/qq_36584673/article/details/144310505)
56. [（**CVPR 2022 Oral**）【图像去噪】论文精读：**MAXIM**: Multi-Axis MLP for Image Processing](https://blog.csdn.net/qq_36584673/article/details/142812557)
57. [（**NeurlPS 2021**）【图像去噪】论文精读：Learning to Generate Realistic Noisy Images via Pixel-level Noise-aware Adversarial Training（**PNGAN**）](https://blog.csdn.net/qq_36584673/article/details/142812343)
58. [（**NeurlPS 2021**）【图像去噪】论文复现：生成逼真噪声图像！通用方法可用于其他模型微调涨点！**PNGAN**的Pytorch版本源码复现，详解PNGAN网络结构，清晰易懂，从源码理解论文公式！](https://blog.csdn.net/qq_36584673/article/details/143354544)
59. [（**CVPR 2022**）【图像去噪】论文精读：**Uformer**: A General U-Shaped Transformer for Image Restoration](https://blog.csdn.net/qq_36584673/article/details/142819137)
60. [（**CVPR 2022**）【图像去噪】论文复现：Modulator助力Transformer块校准特征！预训练模型可以下载啦！**Uformer**的Pytorch源码复现，图文结合全流程详细复现，源码详细注释，思路清晰明了！](https://blog.csdn.net/qq_36584673/article/details/141672132)
61. [（**CVPR 2021**）【图像去噪】论文精读：**HINet**: Half Instance Normalization Network for Image Restoration](https://blog.csdn.net/qq_36584673/article/details/142287112)
62. [（**CVPR 2021**）【图像去噪】论文复现：比赛夺魁！半实例归一化网络**HINet**的Pytorch源码复现，图文结合手把手复现，轻松跑通，HINet结构拆解与代码实现，结构图与代码变量一一对应，逐行注释！](https://blog.csdn.net/qq_36584673/article/details/142895081)
63. [（**CVPR 2022 Oral**）【图像去噪】论文精读：**Restormer**: Efficient Transformer for High-Resolution Image Restoration](https://blog.csdn.net/qq_36584673/article/details/142452548)
64. [（**CVPR 2022 Oral**）【图像去噪】论文复现：CVPR 2022 oral！**Restormer**的Pytorch源码复现，跑通训练和测试源码，报错改进全记录，由大到小拆解网络结构，由小到大实现模型组合，代码逐行注释！](https://blog.csdn.net/qq_36584673/article/details/142869271)
65. [（**ECCV 2022**）【图像去噪】论文精读：Simple Baselines for Image Restoration（**NAFNet**）](https://blog.csdn.net/qq_36584673/article/details/142282634)
66. [（**ECCV 2022**）【图像去噪】论文复现：大道至简！**NAFNet**的Pytorch源码复现！跑通NAFNet源码，补充PlainNet，由大到小拆解NAFNet网络结构，由小到大实现结构组合，逐行注释！](https://blog.csdn.net/qq_36584673/article/details/142282656)
67. [（**TIP 2024**）【图像去噪】论文精读：Single Stage Adaptive Multi-Attention Network for Image Restoration（**SSAMAN**）](https://blog.csdn.net/qq_36584673/article/details/142354046)
68. [【图像去噪】论文精读：**KBNet**: Kernel Basis Network for Image Restoration](https://blog.csdn.net/qq_36584673/article/details/142352973)
69. [【图像去噪】论文复现：补充KBNet训练过程！**KBNet**的Pytorch源码复现，全流程跑通代码，模型结构详细拆解，逐行注释！](https://blog.csdn.net/qq_36584673/article/details/142432179)
70. [（**TMLR 2024**）【图像去噪】论文精读：CascadedGaze: Efficiency in Global Context Extraction for Image Restoration（**CGNet**）](https://blog.csdn.net/qq_36584673/article/details/142353110)
71. [（**TMLR 2024**）【图像去噪】论文复现：全网首发独家！**CGNet**的Pytorch源码复现，全流程跑通代码，训练40万次迭代，模型结构详细拆解，逐行注释，附训练好的模型文件，免费下载！](https://blog.csdn.net/qq_36584673/article/details/142458668)
72. [（**PR 2024**）【图像去噪】论文精读：Dual Residual Attention Network for Image Denoising（**DRANet**）](https://blog.csdn.net/qq_36584673/article/details/143399790)
73. [（**PR 2024**）【图像去噪】论文复现：研究生发SCI范例！**DRANet**的Pytorch源码复现，高斯去噪和真实噪声去噪全流程详解，模型结构示意图与代码变量一一对应，注释详尽，新手友好，单卡可跑！](https://blog.csdn.net/qq_36584673/article/details/143429780)
74. [（**Multimedia Systems 2024**）【图像去噪】论文精读：Dual convolutional neural network with attention for image blind denoising（**DCANet**）](https://blog.csdn.net/qq_36584673/article/details/143742259)
75. [（**Multimedia Systems 2024**）【图像去噪】论文复现：第一个双CNN+双注意力的去噪模型！**DCANet**的Pytorch源码复现，高斯去噪和真实噪声去噪全流程详解，模型结构示意图与代码变量一一对应，注释详尽，新手友好，单卡可跑！](https://blog.csdn.net/qq_36584673/article/details/143760983)
76. [（**ECCV 2024**）【图像去噪】论文精读：**DualDn**: Dual-domain Denoising via Differentiable ISP](https://blog.csdn.net/qq_36584673/article/details/144289496)
77. [（**ECCV 2024**）【图像去噪】论文精读：**MambaIR**: A Simple Baseline for Image Restoration with State-Space Model](https://shixiaoda.blog.csdn.net/article/details/142282678)
78. [（**ECCV 2024**）【图像去噪】论文复现：Man！what can I say？**MambaIR**的Pytorch源码复现，跑通全流程，轻松解决环境配置问题，图文结合按步骤执行傻瓜式教程，由大到小拆解网络，由小到大实现组合！](https://shixiaoda.blog.csdn.net/article/details/143103850)
79. [【图像去噪】论文精读：**Restore-RWKV**: Efficient and Effective Medical Image Restoration with RWKV](https://shixiaoda.blog.csdn.net/article/details/144523730)
## 半监督、无监督、自监督、扩散模型（本质也是无监督）
1. [（**CVPR 2018**）【图像去噪】论文精读：Deep Image Prior（**DIP**）](https://blog.csdn.net/qq_36584673/article/details/143974947)
2. [（**CVPR 2018**）【图像去噪】论文复现：扩散模型思想鼻祖！**DIP**的Pytorch源码复现，执行教程，代码解析，注释详细，只需修改图像路径即可测试自己的噪声图像！](https://blog.csdn.net/qq_36584673/article/details/144073144)
3. [（**ICML 2018**）【图像去噪】论文精读：**Noise2Noise**: Learning Image Restoration without Clean Data（**N2N**）](https://blog.csdn.net/qq_36584673/article/details/141828843)
4. [（**ICML 2018**）【图像去噪】论文复现：倒反天罡！老思想新创意，无需Ground-truth！Pytorch实现无监督图像去噪开山之作**Noise2Noise**！附训练好的模型文件！](https://blog.csdn.net/qq_36584673/article/details/141957263)
5. [（**CVPR 2019**）【图像去噪】论文精读：**Noise2Void**-Learning Denoising from Single Noisy Images（**N2V**）](https://blog.csdn.net/qq_36584673/article/details/141996052)
6. [（**CVPR 2029**）【图像去噪】论文复现：降维打击！图像对输入变成像素对输入！Pytorch实现**Noise2Void（N2V）**，基于U-Net模型训练，简洁明了理解N2V核心思想！附训练好的灰度图和RGB图的模型文件！](https://blog.csdn.net/qq_36584673/article/details/141996345)
7. [（**PMLR 2019**）【图像去噪】论文精读：Noise2Self: Blind Denoising by Self-Supervision（**N2S**）](https://blog.csdn.net/qq_36584673/article/details/144212546)
8. [（**PMLR 2019**）【图像去噪】论文复现：自监督盲去噪！Noise2Self（**N2S**）的Pytorch源码复现，跑通源码，测试单图像去噪，解决了代码中老版本存在的问题，mask核心代码解析，注释清晰易懂！](https://blog.csdn.net/qq_36584673/article/details/144290449)
9. [（**CVPR 2020**）【图像去噪】论文精读：Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image（**S2S**）](https://blog.csdn.net/qq_36584673/article/details/144242137)
10. [（**CVPR 2020**）【图像去噪】论文复现：单噪声图像输入的自监督图像去噪！**Self2Self(S2S)** 的Pytorch版本源码复现，跑通代码，原理详解，代码实现、网络结构、论文公式相互对应，注释清晰，附修改后的完整代码！](https://blog.csdn.net/qq_36584673/article/details/144281526)
11. [（**CVPR 2021**）【图像去噪】论文精读：Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising（**R2R**）](https://blog.csdn.net/qq_36584673/article/details/144375305)
12. [（**CVPR 2021**）【图像去噪】论文复现：一步破坏操作提升自监督去噪性能！**R2R**的Pytorch源码复现，跑通源码，获得评估指标和去噪结果，核心原理与代码详解，双系统单卡可跑，附训练好的模型文件！](https://blog.csdn.net/qq_36584673/article/details/144401628)
13. [（**CVPR 2021**）【图像去噪】论文精读：**Neighbor2Neighbor**: Self-Supervised Denoising from Single Noisy Images](https://blog.csdn.net/qq_36584673/article/details/144375367)
14. [（**CVPR 2021**）【图像去噪】论文复现：相邻像素子图采样助力自监督去噪学习！**Neighbor2Neighbor**的Pytorch源码复现，跑通及补充测试代码，获得去噪结果和PSNR/SSIM，与论文中基本一致，单卡可跑！](https://blog.csdn.net/qq_36584673/article/details/144510619)
15. [（**ECCV 2020**）【图像去噪】论文精读：Unpaired Learning of Deep Image Denoising（**DBSN**）](https://blog.csdn.net/qq_36584673/article/details/144537012)
16. [（**ECCV 2020**）【图像去噪】论文复现：膨胀卷积助力盲点网络自监督训练！**DBSN**的Pytorch源码复现，跑通源码，补充源码中未提供的保存去噪结果图像代码，获得PSNR/SSIM，原理公式示意图与代码对应！](https://blog.csdn.net/qq_36584673/article/details/144537031)
17. [（**CVPR 2022**）【图像去噪】论文精读：**AP-BSN**: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot](https://blog.csdn.net/qq_36584673/article/details/144583771)
18. [（**CVPR 2022**）【图像去噪】论文复现：非对称PD和R3后处理助力自监督盲点网络去噪！**AP-BSN**的Pytorch源码复现，跑通源码，获得结果，可作为实验对比方法，结构图与源码实现相对应，轻松理解，注释详细！](https://blog.csdn.net/qq_36584673/article/details/144583778)
19. [（**CVPR 2023**）【图像去噪】论文精读：Zero-Shot Noise2Noise: Efficient Image Denoising without any Data（**ZS-N2N**）](https://blog.csdn.net/qq_36584673/article/details/144644157)
20. [（**CVPR 2023**）【图像去噪】论文复现：大道至简！**ZS-N2N**的Pytorch源码复现，跑通源码，获得指标计算结果，补充保存去噪结果图像代码，代码实现与论文理论对应！](https://blog.csdn.net/qq_36584673/article/details/144718166)
21. [（**CVPR 2023**）【图像去噪】论文精读：Spatially Adaptive Self-Supervised Learning for Real-World Image Denoising（**BNN-LAN**）](https://shixiaoda.blog.csdn.net/article/details/144766419)
22. [（**CVPR 2023**）【图像去噪】论文复现：分区域提升盲点网络性能！**BNN-LAN**的Pytorch源码复现，跑通源码，获得指标计算结果，补充保存去噪结果图像代码，代码实现与理论公式一一对应！](https://shixiaoda.blog.csdn.net/article/details/144879889)
23. [（**CVPR 2023**）【图像去噪】论文精读：**LG-BPN**: Local and Global Blind-Patch Network for Self-Supervised Real-World Denoising](https://shixiaoda.blog.csdn.net/article/details/144766430)
24. [（**ECCV 2024**）【图像去噪】论文精读：Asymmetric Mask Scheme for Self-Supervised Real Image Denoising（**AMSNet**）](https://blog.csdn.net/qq_36584673/article/details/144289366)
25. [（**TPAMI 2024**）【图像去噪】论文精读：Stimulating Diffusion Model for Image Denoising via Adaptive Embedding and Ensembling（**DMID**）](https://blog.csdn.net/qq_36584673/article/details/143830392)
26. [（**TPAMI 2024**）【图像去噪】论文复现：扩散模型用于图像去噪！**DMID**的Pytorch源码复现，跑通测试流程，结构梳理和拆解，理论公式与源码对应，注释详细， Window和Linux下单卡均可运行！](https://blog.csdn.net/qq_36584673/article/details/143908946)
27. [（**CVPR 2024**）【图像去噪】论文精读：**LAN**: Learning to Adapt Noise for Image Denoising](https://blog.csdn.net/qq_36584673/article/details/144644074)
28. [（**AAAI 2025**）【图像去噪】论文精读：Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising（**TBSN**）](https://shixiaoda.blog.csdn.net/article/details/144931866)
29. [（**AAAI 2025**）【图像去噪】论文复现：Transfomer块增大自监督去噪盲点网络感受野！**TBSN**的Pytorch源码复现，跑通全流程，获取指标计算结果，补充保存图像代码，模型结构示意图与源码实现一一对应，思路清晰！](https://shixiaoda.blog.csdn.net/article/details/144962904)

## OOD泛化
1. [（**CVPR 2023**）【图像去噪】论文精读：Masked Image Training for Generalizable Deep Image Denoising（**MaskedDenoising**）](https://blog.csdn.net/qq_36584673/article/details/144179608)
1. [（**CVPR 2024**）【图像去噪】论文精读：Robust Image Denoising through Adversarial Frequency Mixup（**AFM**）](https://blog.csdn.net/qq_36584673/article/details/142479349)
2. [（**CVPR 2024**）【图像去噪】论文复现：提高真实噪声去噪模型的泛化性！**AFM**的Pytorch源码复现，跑通AFM源码全流程，图文结合，网络结构拆解，模块对应源码注释，源码与论文公式对应！](https://blog.csdn.net/qq_36584673/article/details/143623998)
3. [（**CVPR 2024**）【图像去噪】论文精读：Transfer CLIP for Generalizable Image Denoising（**CLIPDenoising**）](https://blog.csdn.net/qq_36584673/article/details/143816117)
4. [（**CVPR 2024**）【图像去噪】论文复现：CLIP用于图像去噪提升泛化性！**CLIPDenoising**的Pytorch源码复现，跑通CLIPDenoising全流程，图文结合，网络结构梳理和拆解，对应源码注释！](https://blog.csdn.net/qq_36584673/article/details/143827052)
## 其他
1. [【图像去噪】实用小技巧 | 使用matlab将.mat格式的图像转成.png格式的图像，适用于DnD数据集的转换，附DND图像形式的数据集](https://blog.csdn.net/qq_36584673/article/details/141848839)
2. [【图像去噪】实用小技巧 | 使用matlab将.mat格式的SIDD验证集转成.png格式的图像块，附图像形式的SIDD验证集](https://blog.csdn.net/qq_36584673/article/details/142322906)

# 资料汇总（持续更新中。。。）
数据集：

1. BSD：[https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
2. CBSD68：[https://github.com/clausmichele/CBSD68-dataset](https://github.com/clausmichele/CBSD68-dataset)
3. Nam：[http://snam.ml/research/ccnoise](http://snam.ml/research/ccnoise)（已失效，可以从[https://github.com/csjunxu/MCWNNM-ICCV2017](https://github.com/csjunxu/MCWNNM-ICCV2017)中找到）
4. 汇总：[https://blog.csdn.net/iteapoy/article/details/86062640](https://blog.csdn.net/iteapoy/article/details/86062640)
5. DND：[https://noise.visinf.tu-darmstadt.de/](https://noise.visinf.tu-darmstadt.de/)
6. McMaster：[https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm)
7. SIDD：[https://abdokamel.github.io/sidd/](https://abdokamel.github.io/sidd/)
8. PolyU：[https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset)
9. DIV2K：[https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
10. RENOIR：[https://ani.stat.fsu.edu/~abarbu/Renoir.html](https://ani.stat.fsu.edu/~abarbu/Renoir.html) 
11. NIND：[https://commons.wikimedia.org/wiki/Natural_Image_Noise_Dataset](https://commons.wikimedia.org/wiki/Natural_Image_Noise_Dataset)
12. ImageNet验证集：[https://image-net.org/challenges/LSVRC/2012/2012-downloads.php](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)
13. AAPM数据集（CT图像）：[https://www.aapm.org/GrandChallenge/LowDoseCT/](https://www.aapm.org/GrandChallenge/LowDoseCT/)

<font color = 'red'>**注：数据集耐心点搜都能找到，优先官方下载，有的官方失效了在Github上找（可能在某个论文的源码中）。**</font>
# 问题汇总（持续更新中。。。）
1. 没有图像处理基础、深度学习基础、代码能力弱、英语也不太好能看懂专栏内文章吗？
**答：可以，但初期会比较吃力，先复现几篇简单的文章，全力搞懂代码，多学多积累很快就能上手。**
2. xxx型号的GPU xxG显存够用吗？
**答：根据你的实际情况配置GPU，实验室什么条件就用什么，实验室提供不了选择租卡比较划算，因为可以根据模型大小、数据集大小、源码参数设置等灵活选择卡数。不建议个人配置Windows显卡，因为大多数论文的实验环境都是Linux，而且单卡在模型复杂度较大时显存会不够用。**

# 公众号
关注下方公众号【十小大的底层视觉工坊】，公众号将更新精炼版论文，帮助你用碎片化时间快速掌握论文核心内容。

**关注公众号可免费领取一份200+即插即用模块资料，领取方式关注后见自动回复！**
