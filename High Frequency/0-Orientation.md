medium题目要在20分钟内做完，并且跑3次以内的test case

LinkedIn, handshake, 海投网
10%获取面试的概率

450-500
每个project写4-5个bullet points
过去式描述已完成的Projects，单句写完整，分句要注意，语言具体而简洁

没有实习：编/加工
实习弱：
实习不相关：

backgroud check: 

中小型公司：unpaid 类型
contact reference

至少2项实习经历

BQ讲经历：故事的方式
基于兴趣选择实习，on board发现有gap，用一周所有时间快速学习公司业务和软件，开发的目的是什么，遇到了什么棘手的问题，先尝试了什么方法，特别焦虑着急，都在想如何突破，突然有一天睡觉前总结内容发现了灵感，这样最challenge的问题解决了，接下来就一马平川，最后在实习结束的1周半之前就完成了主要的任务
方法：答如所问，给出例子和场景，产生共情心理，最后要happy ending
负面问题：自己的兴趣和同事的兴趣，做一个算法的deep dive造成时间比预期晚了3天时间，最后整个项目晚了一周，复盘发现一些其他也有些dependency，最后learn a lesson：以后遇到这种情况，不能用工作时间来explore深度的东西，我意识到我的delivery time是非常重要的。应该在spare time来学，并在spare time给同事讲他们想要获取的信息。在之后的实习过程自己都是这么做的，效果也很好。之后我都会提前个1周时间来完成自己的工作内容
例子准备8-12个

Cover Letter：第二简历，一定要match岗位要求
要写，通过模板

如果公司提供Sponsorship -> Yes
不提供Sponsorship -> No

uscis h1b sponsor employer
green card eb5 半年时间

Why us?
宏观：产品、远景、文化等欣赏
个人：重点在于技能匹配
发展：与未来发展规划契合：就是想学习，技术栈也喜欢，在这个岗位至少有3-5年成长期，之后如果有更强的能力，open to更多的responsibility

No PAIN. NO GAIN.
努力+认真=Dream offer

# 时间复杂度整理
n≤30, 指数级别, dfs+剪枝，状态压缩dp
n≤100 => O(n^3)，floyd，dp，高斯消元
n≤1000 => O(n^2)，O(n^2logn)，dp，二分，朴素版Dijkstra、朴素版Prim、Bellman-Ford
n≤10^4 => O(n∗√n)，块状链表、分块、莫队
n≤10^5 => O(nlogn) => 各种sort，线段树、树状数组、set/map、heap、拓扑排序、dijkstra+heap、prim+heap、Kruskal、spfa、求凸包、求半平面交、二分、CDQ分治、整体二分、后缀数组、树链剖分、动态树
n≤10^6 => O(n), 以及常数较小的 O(nlogn)O(nlogn) 算法 => 单调队列、 hash、双指针扫描、并查集，kmp、AC自动机，常数比较小的 O(nlogn)O(nlogn) 的做法：sort、树状数组、heap、dijkstra、spfa
n≤10^7 => O(n)，双指针扫描、kmp、AC自动机、线性筛素数
n≤10^9 => O(√n)，判断质数
n≤10^18 => O(logn)，最大公约数，快速幂，数位DP
