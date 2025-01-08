# image-denoising
图像去噪专栏
【图像去噪】专栏链接：

​blog.csdn.net/qq_36584673/category_12756147.html

# 前言 
请读者先思考几个问题：
 你是否在全网苦寻【图像去噪（Image Denoising）】的相关资料？
 你的目标是否是看懂【图像去噪（Image Denoising）】的相关论文，复现代码，跑出结果，并试图创新？
 你是否需要发表【图像去噪（Image Denoising）】的相关论文毕业？
 你是否需要做【图像去噪（Image Denoising）】的相关项目，开发软件，研究算法，获得专利或者软著？
 ......
只要你的需求与【图像去噪（Image Denoising）】有关，那么请继续往下沉浸阅读。
以刚读研的朋友为例，进组后导师让你研究与【图像去噪（Image Denoising）】的相关课题，但又没有时间指导你，只能自己研究。
这个时候你该怎么办呢？刚入门的小白，什么也不了解，只能是自己搜集资料或者问师兄师姐。
如果你问师兄师姐，怎么搞科研啊？如何看懂论文啊？如何读懂复杂的代码啊？
他们大概率会回复你：硬啃。一句一句地读论文呢，一行一行地理解代码。无他，唯手熟尔。
你摇了摇头，觉得他们“自私”，没有把“看家本领”传授于你。于是，你开始像无头苍蝇一样，试图寻找好的学习方法，希望通过一种好的学习方法来提升科研效率。
想法很好，但浪费时间和精力，而且没有开始真正的学习，只是在学习的边缘徘徊。其实，就差一步，管它科研难不难，学就完事了。
学习无捷径，大道至简，先学起来，再慢慢修正学习路线，最后到达顶峰。
很幸运你看到了本文，本专栏的意义就在于用我走过的路，带你看懂论文，读懂代码，实现论文复现，避免走进“找方法找资料”等费时费力的弯路。领先实验室同门，早发论文，早做项目，受导师青睐，尽早达到毕业要求去实习。
三年时间转瞬即逝，必须马上开始！而你要做的，只是认认真真的读完每一篇专栏内的文章即可，相信你一定会有收获，科研能力越来越强，顺利完成你的目标。
# 适配人群
看不懂论文、写不出代码、无创新思路的新手小白，希望快速入门
课题与去噪相关，需要发论文毕业的本科生和研究生（硕士、博士），缺乏科研能力以及论文写作经验的实验室难兄难弟
导师放养，不知道选择哪个方向入手的迷茫者
 需要论文、专利、软著等评职称的相关人员
 有做去噪相关项目的本科生、研究生（硕士、博士）、企业开发人员、算法工程师等
# 专栏简介
专栏名为【Pytorch深度学习图像去噪算法100例（论文精读+复现）】。顾名思义，共三点限制：
1. 专栏内涉及的算法都是基于深度学习的图像去噪算法。
2. 所有算法都是基于Pytorch框架实现的。论文公开的源码是Pytorch的我们就基于源码复现，不是用Pytorch框架实现或者未公开源码的算法，我会用Pytorch重新复现一遍，以达成代码的统一，方便大家做实验。
3. 图像去噪领域顶刊顶会文章精选100篇论文，复现100个源码（目前更新中，等达成后再继续收录）。
专栏内文章主要为两部分：【论文精读】与【论文复现】
 论文精读：读懂论文，总结提炼，聚焦核心内容，不只是全文翻译
 论文复现：跑通流程，源码解析，提升代码能力，得到去噪结果以及指标计算
综合而言，从大到小拆解模型结构，从小到大实现模型搭建。实现论文与源码的统一，深入理解论文行文逻辑与代码实现逻辑，融汇贯通二者思想，并学以致用。
更具体详细的内容见【阅读方法】小节。
# 专栏亮点
省时：图像去噪领域论文全覆盖，算法模型全，主流的算法模型都会涉及结合能搜集到的资料深入浅出解读，按【文章目录】顺序阅读即可，节省搜论文的时间。【论文复现】文章末尾会提供训练好各放大倍数下的最优性能（PSNR/SSIM）模型权重文件（若源码已提供则没有），读者可以下载直接使用，方便论文中实验部分方法对比（视觉效果和量化指标）或直接应用模型到自己的图像数据上。
省力：大白话讲解，手把手教学，带你读懂论文，跑通代码，提升代码能力，理解文章的算法和创新点，避免一个人对着论文死磕但还是不理解的情况出现。按部就班，肯定能跑出好的模型结果，帮助你了解论文内容和结构，积少成多，学会写论文。同时，让你少走弯路，避免模型调参四处碰壁，涉及的模型都提供训练好的权重文件，可以直接拿来测试自己的图像。
省事：努力做全网最全最好最详细的【图像去噪】专栏，专栏永久更新，第一时间更新最新的图像去噪领域论文，关注订阅后一劳永逸，提供免费答疑及交流。
阅读方法
对于【论文精读】文章，请对照原文，除了本身的文章翻译外，重点阅读红色加粗部分，其为该小节的精炼总结。如果想略读文章，只追求最精炼的文章核心内容，则关注本文末尾的微信公众号免费查看。
对于【论文复现】文章，首先，跑通源码，每篇复现文章的第一部分均是详细的跑通步骤，包括设置配置文件、设置数据集和预训练模型路径、需要的依赖库、报错解决等。其次，根据需要阅读相应的源码部分，如数据预处理、网络结构实现、训练、测试、其他补充代码，重点在于与论文直接相关的网络结构以及特殊损失函数的实现。最后，总结与反思，复盘复现过程，包含核心内容（可能是某个即插即用模块需要积累）、可能的改进方向、需要积累的代码等。

SIDD数据集上一些SOTA方法复现的去噪可视化结果
# 文章目录
图像去噪任务大体分为监督和非监督两类，监督类模型数据集中包含Ground-Truth，非监督一般没有Ground-Truth。
文章目录按年份和模型性能综合排序，按顺序阅读即可。
## 有监督
（TIP 2017）【图像去噪】论文精读：Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising（DnCNN）
（TIP 2017）【图像去噪】论文复现：新手入门必看！DnCNN的Pytorch源码训练测试全流程解析！为源码做详细注释！补充DnCNN-B和DnCNN-3的模型训练和测试！附各种情况下训练好的模型权重文件！
（TIP 2018）【图像去噪】论文精读：FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising
（TIP 2018）【图像去噪】论文复现：适合新手小白的Pytorch版本FFDNet复现！详解FFDNet源码！数据处理、模型训练和验证、推理测试全流程讲解！新手小白都能看懂，学习阅读毫无压力，去噪入门必看！
（CAAI Transactions on Intelligence Technology 2019）【图像去噪】论文精读：Enhanced CNN for image denoising（ECNDNet）
（CAAI Transactions on Intelligence Technology 2019）【图像去噪】论文复现：Pytorch实现ECNDNet！入门项目，适合新手小白学习使用，Windows和Linux下均可运行！附训练好的模型文件，可直接得到去噪结果和指标计算！
（CVPR 2019）【图像去噪】论文精读：Toward Convolutional Blind Denoising of Real Photographs（CBDNet）
（CVPR 2019）【图像去噪】论文复现：适合新手小白的Pytorch版本CBDNet复现！轻松跑通训练和测试代码！简单修改路径即可训练自己的数据集！代码详细注释！数据处理、模型训练和验证、推理测试全流程讲解！
（ICCV 2019）【图像去噪】论文精读：Real Image Denoising with Feature Attention（RIDNet）
（ICCV 2019）【图像去噪】论文复现：适合新手小白的Pytorch版本RIDNet复现！轻松跑通训练和测试代码！RIDNet网络结构实现拆解！简单修改路径即可训练自己的数据集！模型训练推理测试全流程讲解！
（NeurIPS 2016）【图像去噪】论文精读：Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections（REDNet）
（NeurIPS 2016）【图像去噪】论文复现：Pytorch实现REDNet的三种结构！简单修改路径即可跑通全部代码并训练自己的数据集！支持灰度图和RGB图训练！附训练好的模型文件可直接测试图像得到去噪结果以及评价指标！ 
（ICCV 2017）【图像去噪】论文精读：MemNet: A Persistent Memory Network for Image Restoration
（ICCV 2017）【图像去噪】论文复现：全网最细的Pytorch版本实现MemNet！论文中的网络结构图与代码中的每个变量一一对应！实现思路一目了然！附完整代码和训练好的模型权重文件！
（CVPR 2018）【图像去噪】论文精读：xUnit: Learning a Spatial Activation Function for Efficient Image Restoration
（CVPR 2018）【图像去噪】论文复现：代替ReLU！Pytorch实现即插即用激活函数模块xUnit，并插入到DnCNN中实现xDnCNN！
（CVPR 2018）【图像去噪】论文精读：Multi-level Wavelet-CNN for Image Restoration（MWCNN）
（CVPR 2018）【图像去噪】论文复现：小波变换替代上下采样！Pytorch实现MWCNN，数据处理、模型训练和验证、推理测试全流程讲解，无论是科研还是应用，新手小白都能看懂，学习阅读毫无压力，去噪入门必看！
（CVPR 2017）【图像去噪】论文精读：Learning Deep CNN Denoiser Prior for Image Restoration（IRCNN）
（CVPR 2017）【图像去噪】论文复现：支持任意大小的图像输入！四十多行实现Pytorch极简版本的IRCNN，各种参数和测试集平均PSNR结果与论文一致！
（ECCV 2020）【图像去噪】论文精读：Spatial-Adaptive Network for Single Image Denoising（SADNet）
（ECCV 2020）【图像去噪】论文复现：三万字长文详解SADNet的Pytorch源码！全网最详细保姆级傻瓜式教程，新手小白也能看懂，代码逐行注释，跑通代码得到去噪结果毫无压力！网络结构图与模型定义的量一一对应！
（ICLR 2019）【图像去噪】论文精读：Residual Non-local Attention Networks for Image Restoration（RNAN）
（ICLR 2019）【图像去噪】论文复现：非局部注意力机制提升去噪性能！Pytorch实现RNAN，解决out of memory问题，论文中结构图与代码变量一一对应，清晰明了保证看懂！附训练好的模型文件！
（CVPR 2020）【图像去噪】论文精读：Transfer Learning from Synthetic to Real-Noise Denoising with Adaptive Instance  Normalization（AINDNet）
（NN 2020）【图像去噪】论文精读：Attention-guided CNN for Image Denoising（ADNet）
（NN 2020）【图像去噪】论文复现：注意力机制助力图像去噪！ADNet的Pytorch源码复现，跑通全流程源码，补充源码中未提供的保存去噪结果图像代码，ADNet网络结构图与源码对应，新手友好，单卡可跑！
（NeurIPS 2019）【图像去噪】论文精读：Variational Denoising Network: Toward Blind Noise Modeling and Removal（VDNet）
（NeurIPS 2019）【图像去噪】论文复现：包能看懂！VDNet的Pytorch源码全解析！逐行详细注释，理论与代码结合，提升代码能力！
（TPAMI 2021）【图像去噪】论文精读：Plug-and-Play Image Restoration with Deep Denoiser Prior（DRUNet）
（TPAMI 2021）【图像去噪】论文复现：无偏置在超大噪声下也能有效去噪！DRUNet的Pytorch源码复现，跑通DRUNet源码，得到去噪结果和评价指标，可作为实验中的对比方法，源码结构梳理，注释清晰，单卡可运行！
（CVPR 2021）【图像去噪】论文精读：SwinIR: Image Restoration Using Swin Transformer
（CVPR 2021）【图像去噪】论文复现：Swin Transfomer用于图像恢复！SwinIR的Pytorch源码复现，跑通源码，测试高斯去噪，使用预训练模型得到PSNR/SSIM以及去噪后图像！
（Machine Intelligence Research 2023）【图像去噪】论文精读：Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis（SCUNet）
（Machine Intelligence Research 2023）【图像去噪】论文复现：Swin Transformer块助力图像去噪！SCUNet的Pytorch源码复现，跑通SCUNet源码，理论结构梳理，获得SCUNet去噪结果，可作为实验中的对比方法！
（ECCV 2020）【图像去噪】论文精读：Dual Adversarial Network: Toward Real-world Noise Removal and Noise Generation（DANet）
（ECCV 2020）【图像去噪】论文复现：双对抗网络去除和生成逼真噪声！DANet的Pytorch源码复现，用于去噪和生成逼真噪声，训练、测试、损失函数、整体架构逻辑详解，图文结合，源码注释详细，单卡可跑！
（ECCV 2020）【图像去噪】论文精读：Learning Enriched Features for Real Image Restoration and Enhancement（MIRNet）
（ECCV 2020）【图像去噪】论文复现：全网最细！MIRNet的Pytorch源码复现全记录！论文中模型结构图与代码变量一一对应，保证看懂！踩坑报错复盘，一一排雷，Windows下也能轻松运行，代码逐行注释！
（CVPR 2021）【图像去噪】论文精读：Multi-Stage Progressive Image Restoration（MPRNet）
（CVPR 2021）【图像去噪】论文复现：三阶段网络！MPRNet的Pytorch源码复现，跑通全流程，图文结合手把手教学，由大到小拆解网络结构，由小到大实现结构组合，结构示意图与代码变量一一对应，全部源码逐行注释！
（CVPR 2021）【图像去噪】论文精读：NBNet: Noise Basis Learning for Image Denoising with Subspace Projection
（CVPR 2021）【图像去噪】论文复现：子空间投影提升去噪效果！NBNet的Pytorch源码复现，跑通代码，源码详解，注释详细，图文结合，补充源码中缺少的“保存去噪后的结果图像”代码，指出源码中模型实现的潜在问题
（EUSIPCO 2022）【图像去噪】论文精读：SELECTIVE RESIDUAL M-NET FOR REAL IMAGE DENOISING（SRMNet）
（EUSIPCO 2022）【图像去噪】论文复现：新手友好！SRMNet的Pytorch源码复现，跑通全流程源码，详细步骤，图文说明，注释详细，模型拆解示意图与代码一一对应，包能看懂，Windows和Linux下均可运行！
（CVPR 2021）【图像去噪】论文精读：Pre-Trained Image Processing Transformer（IPT）
（CVPR 2021）【图像去噪】论文复现：底层视觉通用预训练Transformer！IPT的Pytorch源码复现， 跑通IPT源码，获得测试集上平均PSNR和去噪结果，可作为实验中的对比方法！
（ISCAS 2022）【图像去噪】论文精读：SUNet: Swin Transformer UNet for Image Denoising
（ISCAS 2022）【图像去噪】论文复现：Swin Transformer用于图像去噪！SUNet的Pytorch源码复现，训练，测试高斯噪声图像流程详解，计算PSNR/SSIM，Windows和Linux下单卡可跑！
（CVPR 2020 Oral）【图像去噪】论文精读：CycleISP: Real Image Restoration via Improved Data Synthesis
（CVPR 2020 Oral）【图像去噪】论文复现：制作真实噪声图像数据集！CycleISP的Pytorch源码复现，跑通流程，可以用于制作自己的真实噪声图像数据集，由大到小拆解CycleISP网络结构，由小到大实现结构组合！
（CVPR 2021 Oral）【图像去噪】论文精读：Adaptive Consistency Prior based Deep Network for Image Denoising（DeamNet）
（CVPR 2021 Oral）【图像去噪】论文复现：扎实理论提升去噪模型的可解释性！DeamNet的Pytorch源码复现，跑通全流程源码，模型结构图与源码对应拆解，补充源码中没有的保存图像代码！
（AAAI 2020）【图像去噪】论文精读：When AWGN-based Denoiser Meets Real Noises（PD-Denoising）
（AAAI 2020）【图像去噪】论文复现：用Pixel-shuffle构建高斯噪声与真实噪声之间的联系！PD-Denoising源码复现，跑通训练和测试代码，可得到去噪结果和PSNR，理论公式与源码对应，图文结合详解！
（CVPR 2022 Oral）【图像去噪】论文精读：MAXIM: Multi-Axis MLP for Image Processing
（NeurlPS 2021）【图像去噪】论文精读：Learning to Generate Realistic Noisy Images via Pixel-level Noise-aware Adversarial Training（PNGAN）
（NeurlPS 2021）【图像去噪】论文复现：生成逼真噪声图像！通用方法可用于其他模型微调涨点！PNGAN的Pytorch版本源码复现，详解PNGAN网络结构，清晰易懂，从源码理解论文公式！
（CVPR 2022）【图像去噪】论文精读：Uformer: A General U-Shaped Transformer for Image Restoration
（CVPR 2022）【图像去噪】论文复现：Modulator助力Transformer块校准特征！预训练模型可以下载啦！Uformer的Pytorch源码复现，图文结合全流程详细复现，源码详细注释，思路清晰明了！
（CVPR 2021）【图像去噪】论文精读：HINet: Half Instance Normalization Network for Image Restoration
（CVPR 2021）【图像去噪】论文复现：比赛夺魁！半实例归一化网络HINet的Pytorch源码复现，图文结合手把手复现，轻松跑通，HINet结构拆解与代码实现，结构图与代码变量一一对应，逐行注释！
（CVPR 2022 Oral）【图像去噪】论文精读：Restormer: Efficient Transformer for High-Resolution Image Restoration
（CVPR 2022 Oral）【图像去噪】论文复现：CVPR 2022 oral！Restormer的Pytorch源码复现，跑通训练和测试源码，报错改进全记录，由大到小拆解网络结构，由小到大实现模型组合，代码逐行注释！
（ECCV 2022）【图像去噪】论文精读：Simple Baselines for Image Restoration（NAFNet）
（ECCV 2022）【图像去噪】论文复现：大道至简！NAFNet的Pytorch源码复现！跑通NAFNet源码，补充PlainNet，由大到小拆解NAFNet网络结构，由小到大实现结构组合，逐行注释！
（TIP 2024）【图像去噪】论文精读：Single Stage Adaptive Multi-Attention Network for Image Restoration（SSAMAN）
【图像去噪】论文精读：KBNet: Kernel Basis Network for Image Restoration
【图像去噪】论文复现：补充KBNet训练过程！KBNet的Pytorch源码复现，全流程跑通代码，模型结构详细拆解，逐行注释！
（TMLR 2024）【图像去噪】论文精读：CascadedGaze: Efficiency in Global Context Extraction for Image Restoration（CGNet）
（TMLR 2024）【图像去噪】论文复现：全网首发独家！CGNet的Pytorch源码复现，全流程跑通代码，训练40万次迭代，模型结构详细拆解，逐行注释，附训练好的模型文件，免费下载！
（PR 2024）【图像去噪】论文精读：Dual Residual Attention Network for Image Denoising（DRANet）
（PR 2024）【图像去噪】论文复现：研究生发SCI范例！DRANet的Pytorch源码复现，高斯去噪和真实噪声去噪全流程详解，模型结构示意图与代码变量一一对应，注释详尽，新手友好，单卡可跑！
（Multimedia Systems 2024）【图像去噪】论文精读：Dual convolutional neural network with attention for image blind denoising（DCANet）
（Multimedia Systems 2024）【图像去噪】论文复现：第一个双CNN+双注意力的去噪模型！DCANet的Pytorch源码复现，高斯去噪和真实噪声去噪全流程详解，模型结构示意图与代码变量一一对应，注释详尽，新手友好，单卡可跑！
（ECCV 2024）【图像去噪】论文精读：DualDn: Dual-domain Denoising via Differentiable ISP
（ECCV 2024）【图像去噪】论文精读：MambaIR: A Simple Baseline for Image Restoration with State-Space Model
（ECCV 2024）【图像去噪】论文复现：Man！what can I say？MambaIR的Pytorch源码复现，跑通全流程，轻松解决环境配置问题，图文结合按步骤执行傻瓜式教程，由大到小拆解网络，由小到大实现组合！
【图像去噪】论文精读：Restore-RWKV: Efficient and Effective Medical Image Restoration with RWKV
## 半监督、无监督、自监督、扩散模型（本质也是无监督）
（CVPR 2018）【图像去噪】论文精读：Deep Image Prior（DIP）
（CVPR 2018）【图像去噪】论文复现：扩散模型思想鼻祖！DIP的Pytorch源码复现，执行教程，代码解析，注释详细，只需修改图像路径即可测试自己的噪声图像！
（ICML 2018）【图像去噪】论文精读：Noise2Noise: Learning Image Restoration without Clean Data（N2N）
（ICML 2018）【图像去噪】论文复现：倒反天罡！老思想新创意，无需Ground-truth！Pytorch实现无监督图像去噪开山之作Noise2Noise！附训练好的模型文件！
（CVPR 2019）【图像去噪】论文精读：Noise2Void-Learning Denoising from Single Noisy Images（N2V）
（CVPR 2029）【图像去噪】论文复现：降维打击！图像对输入变成像素对输入！Pytorch实现Noise2Void（N2V），基于U-Net模型训练，简洁明了理解N2V核心思想！附训练好的灰度图和RGB图的模型文件！
（PMLR 2019）【图像去噪】论文精读：Noise2Self: Blind Denoising by Self-Supervision（N2S）
（PMLR 2019）【图像去噪】论文复现：自监督盲去噪！Noise2Self（N2S）的Pytorch源码复现，跑通源码，测试单图像去噪，解决了代码中老版本存在的问题，mask核心代码解析，注释清晰易懂！
（CVPR 2020）【图像去噪】论文精读：Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image（S2S）
（CVPR 2020）【图像去噪】论文复现：单噪声图像输入的自监督图像去噪！Self2Self(S2S) 的Pytorch版本源码复现，跑通代码，原理详解，代码实现、网络结构、论文公式相互对应，注释清晰，附修改后的完整代码！
（CVPR 2021）【图像去噪】论文精读：Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising（R2R）
（CVPR 2021）【图像去噪】论文复现：一步破坏操作提升自监督去噪性能！R2R的Pytorch源码复现，跑通源码，获得评估指标和去噪结果，核心原理与代码详解，双系统单卡可跑，附训练好的模型文件！
（CVPR 2021）【图像去噪】论文精读：Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images
（CVPR 2021）【图像去噪】论文复现：相邻像素子图采样助力自监督去噪学习！Neighbor2Neighbor的Pytorch源码复现，跑通及补充测试代码，获得去噪结果和PSNR/SSIM，与论文中基本一致，单卡可跑！
（ECCV 2020）【图像去噪】论文精读：Unpaired Learning of Deep Image Denoising（DBSN）
（ECCV 2020）【图像去噪】论文复现：膨胀卷积助力盲点网络自监督训练！DBSN的Pytorch源码复现，跑通源码，补充源码中未提供的保存去噪结果图像代码，获得PSNR/SSIM，原理公式示意图与代码对应！
（CVPR 2022）【图像去噪】论文精读：AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot
（CVPR 2022）【图像去噪】论文复现：非对称PD和R3后处理助力自监督盲点网络去噪！AP-BSN的Pytorch源码复现，跑通源码，获得结果，可作为实验对比方法，结构图与源码实现相对应，轻松理解，注释详细！
（CVPR 2023）【图像去噪】论文精读：Zero-Shot Noise2Noise: Efficient Image Denoising without any Data（ZS-N2N）
（CVPR 2023）【图像去噪】论文复现：大道至简！ZS-N2N的Pytorch源码复现，跑通源码，获得指标计算结果，补充保存去噪结果图像代码，代码实现与论文理论对应！
（CVPR 2023）【图像去噪】论文精读：Spatially Adaptive Self-Supervised Learning for Real-World Image Denoising（BNN-LAN）
（CVPR 2023）【图像去噪】论文复现：分区域提升盲点网络性能！BNN-LAN的Pytorch源码复现，跑通源码，获得指标计算结果，补充保存去噪结果图像代码，代码实现与理论公式一一对应！
（CVPR 2023）【图像去噪】论文精读：LG-BPN: Local and Global Blind-Patch Network for Self-Supervised Real-World Denoising
（ECCV 2024）【图像去噪】论文精读：Asymmetric Mask Scheme for Self-Supervised Real Image Denoising（AMSNet）
（TPAMI 2024）【图像去噪】论文精读：Stimulating Diffusion Model for Image Denoising via Adaptive Embedding and Ensembling（DMID）
（TPAMI 2024）【图像去噪】论文复现：扩散模型用于图像去噪！DMID的Pytorch源码复现，跑通测试流程，结构梳理和拆解，理论公式与源码对应，注释详细， Window和Linux下单卡均可运行！
（CVPR 2024）【图像去噪】论文精读：LAN: Learning to Adapt Noise for Image Denoising
（AAAI 2025）【图像去噪】论文精读：Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising（TBSN）
（AAAI 2025）【图像去噪】论文复现：Transfomer块增大自监督去噪盲点网络感受野！TBSN的Pytorch源码复现，跑通全流程，获取指标计算结果，补充保存图像代码，模型结构示意图与源码实现一一对应，思路清晰！
## OOD泛化
（CVPR 2023）【图像去噪】论文精读：Masked Image Training for Generalizable Deep Image Denoising（MaskedDenoising）
（CVPR 2024）【图像去噪】论文精读：Robust Image Denoising through Adversarial Frequency Mixup（AFM）
（CVPR 2024）【图像去噪】论文复现：提高真实噪声去噪模型的泛化性！AFM的Pytorch源码复现，跑通AFM源码全流程，图文结合，网络结构拆解，模块对应源码注释，源码与论文公式对应！
（CVPR 2024）【图像去噪】论文精读：Transfer CLIP for Generalizable Image Denoising（CLIPDenoising）
（CVPR 2024）【图像去噪】论文复现：CLIP用于图像去噪提升泛化性！CLIPDenoising的Pytorch源码复现，跑通CLIPDenoising全流程，图文结合，网络结构梳理和拆解，对应源码注释！
## 其他
【图像去噪】实用小技巧 | 使用matlab将.mat格式的图像转成.png格式的图像，适用于DnD数据集的转换，附DND图像形式的数据集
【图像去噪】实用小技巧 | 使用matlab将.mat格式的SIDD验证集转成.png格式的图像块，附图像形式的SIDD验证集
## 公众号
关注微信公众号【十小大的底层视觉工坊】，更新精炼版论文，帮助你用碎片化时间快速掌握论文核心内容。
关注公众号可免费领取一份200+即插即用模块资料，领取方式关注后见自动回复！
