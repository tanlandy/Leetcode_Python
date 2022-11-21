要求实现一个function take two input （String paragraph, int length）返回 string.  这个paragraph 由一些单词和空格组成，随机选择paragraph中个一个单词， 下一个单词从这个随机选择单词右边接邻单词选，这个右边的单词可能在paragraph中出现不只一次，要求从这些相同的单词里面随机选个单词作为下个单词, 直到单词个数和给定length一样长。. .и
example:
String paragraph = “this is a sentence it is not a good one and it is also bad”
Int length = 5
如果随机选择了sentence作为第一个单词， it 是选择的下一个单词，但是it出现两次，从两个it中随机选择任意一个单词作为下一个单词，直到单词的长度达到length。
Output: sentence it is also bad



Given a string, implement a method that given a word in the sentence, randomly return one of its following words. Plus, if the given wrod is in the end of the string, the first word is counted as
its following word.
解答： 写了一个hashmap， key is every word in the string, value is a list of word that follows it. .1point3acres
Follow up: given a substring of the sentence, return the next words of the substring. ..
解答： 我本来想要用Trie做的， 但是发现 space complexity 太高了， 经面试官提醒， 改成用substring作为hashmap的key。但是implementation 有点麻烦， 特别是包含最后一个word到第一个word的情况， 所有最后也没有写完。

第一道
一串IP检测是不是valid, valid指是否有四个数字部分，数字部分要在[0-255]范围eg:
“12.123.1.213” true
"0..12.324" false
第二道
给一个数字string，它可以组成多少个valid的 IP, 输出
eg:
input
"00123"
output:
"0.0.1.23"
"0.0.12.3"

LC93

1. Random Writer
We'll pick a random start word and then random successive words and output n words in total
text = "this is a sentence it is not a good one and it is also bad"
input: Integer, String
output: String
e.g. input: 5, text
     possible output1: a good one and it
     possible output2: it is a sentence it
说明：随机选择一个单词，然后再随机选择successive word直到满足指定长度
思路：hashmap: String -> List<String> 记录每个单词的所有successive words

2. Follow up:
We'll pick k random start words and then random successive words and output n words in total
text = "this is a sentence it is not a good one and it is also bad"
input: Integer, Integer, String
output: String
e.g. input: k = 2, n = 5, text
     possible output: a good one and it
     possible output: it is not a good
     wrong output: it is a sentence it
说明：首次随机选择连续的 k 个单词，然后每次随机选择这 k 个单词的 successive word
举例：k = 2，首次选择 it is，然后随机选择 not 或者 also，假设选择 also，然后再选择 is also 的 successive word
思路：hashmap: String -‍‌‌‌‌‍‍‍‍‍‌‍‌‍‌‍‌‍‌‌> List<String> 记录每 k 个单词的所有successive words
用StringBuilder作为队列，每次删掉最前面的单词，再加入新单词

第一题：假设有些order还有一些flight，根据order选flight。
每个order会有出发日期和到达日期，flight有出发日期和到达日期，每个flight有限载量
第二题：
类似于这样
map=
1："a" "b"
2: "c"‍‌‌‌‌‍‍‍‍‍‌‍‌‍‌‍‌‍‌‌
get_nums=
11: "aa" "ab" "ba" "bb"
12: "ac" "bc"

题目是面经题 checker board
part1: given a state of the board and a player, get all possible next moves of all checkers of the player
part2: given a move, make a movement and update the board

1. 给你一个webpage和里面有的link (link是另一个webpage), 让你求出通过第一个webpage能访问到的所有webpage的size, webpage里有webpage的size信息Follow up: 给一个有所有node的set, 如何求root webpage
2。check 版本号 输入是两个string num followed by dot followed by int 看哪个更新。follow up：如何validate string
3. Implement hash map 面试官讲了一堆有的没的我还以为要写新几个的hashmap class 同时来override hashmap某些function, 然后分别调用，结果问了半天发现就是implement hashmap
4. 给一个byte[] read（） 让你implement byte read(int size)来read给定size的数据  让你考虑各种可能性follow up: 如果数据是  1 3 2 4 0 1要你decode 成 3 4 4 （要考虑数据很大的情况，不能直接copy，要使用iterator）follow up反过来encode题都不难 但每个题都要分析时间复杂度空间复杂度以及各种corner case和优化

3 opentable 设计，就是设计data model
4 http协议，网站访问过程，bq

LC468 LC93(943)

电面是 这里一道题 checker game
vo四轮
1. project deep dive 需要画 diagram说明
2. coding 是一道oop 这里没有出现过 有一个shipment class有weight这个filed 然后不同weight区间对应不同rate 需要计算一个 shipment的价格这样。。
3.debug  基本上就是string里的特殊字 replace掉 并不难但我脑子仿佛在这里卡住了= = 场面一度十分尴尬

电面是常规ip address的那道题
VO:
1. checker game 没有要求写连吃的operation 提前10分钟就结束了
2. design open table 万年不变
3. 文章随机组成output的那道题 提前写完了 问了个follow up 如果不是一个单词一个单词的拼接的话怎么办 一开始是随机抽一个词然后随机选后面的 现在pass进来一个variable N 这个 N 代表每次要检查的最小subarray的长度 一开始还是随机选一个单词 如果N=3 就直接form从这个词开始长度为3的subarray 然后下次再form的时候需要查这整个subarray后面可能出现的词 加到结果里 每次form的时候都查结果里最后3位之后可能出现的词 这里没见过 不过也不难 写完了之后就开始聊天
4.project deep dive 建议提前熟悉一下做过的project 要draw flow chart 然后这轮一般是hm面 有一票否决权 别的轮都不难 这轮应该是需要着重准备的一轮




简易版lc 17， 假设只有1和2， 我用dfs写的，还是蛮快的。问面试官如果invalid input怎么办，她说throw error, 结果我用java写一时惊慌不记得怎么throw/catch exception就出了点问题。中间还有一个小bug面试官帮我改了，以为可能凉了，但是还是过了。他们的project deep dive是真的问的特别细，一定要知道自己project的components，还有各种algorithms的implementation都是要能解释的。面试前一定要过一遍自己的project。flexport给我面试的人都特别好，推荐！



on campus：
原题 348.  需要俩人模拟玩游戏
所以要写 scanner 输入 和 print 每一次的输出和结果。

然后过了之后是 VO  3小时。
一轮manager的BQ
一轮这里的  换卡的那个题，   eg：  钱包里有2个红色 2个绿色。      想买黑色。   已知黑色可以用2个红色买   问能不能买。   follow up 是discount。   这个题当时就没太明白 所以我也说不太明白 sorry。。。
还有一轮是 这里随机生成文章那个题。  原题是每个单词作为单位看看下一个单词可能是哪一个，  follow up 是 K个单词作为一个整体， 在这个整体的下一个单词做random 选择。



中高难度
沟通max
讲中文
coderpad



