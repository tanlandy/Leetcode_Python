# 06/06/2022
## Intro
the intro has to be up to 2min

## Past Experience 
1. ownership, leadership: tell me about a time when you took on something significant outside your area of reponsibilitiy. why was it important? what was the outcome -> solid example：得了什么任务，任务背景，我采取什么行动，最后结果是如何
   1. What data report look like(what you did speficily)
   2. Example of issue
   3. How to show the result to your teammates -> outcome!!!
2. tell me about a situation that required you to dig deep to get to the root cause? how did you know you were focusing on the right things? what was the outcome? would you have done anything differently? -> issue是什么，如何解决的
   1. 写一个todo list，然后和领导同事来对，确认是对的 -> real example
      1. -> two options: what choosed, why choosed that
   2. Other things that dive deep(challenges)
3. A time strongly disagree with your 
4. co-worker that on something you believe it's important to the business：分歧是什么，自己做了什么说服，最后结果
   1. S: 
   2. A: Data points used 
   3. R: 
5. Learn be curious: describe a time when you took on work outside of your comfort area

## Code
Implement a Map interface, allowing us to put a key-value pair, and retieve it with get(Key) to get lastest, and get(key, time) to get the value at the exact time it was inserted.

example:
put("fruit", "banana") // at time 1
put("fruit", "apple") // at time 2
put("fruit", "orange") // at time 3

get("fruit") => "orange"
get("fruit", "2") => apple

## Feedback:
不要不相关的话，最多5句话，1分钟就够了
一定要clarification

# 02/18/2022
Question1: 
Introduction


What project/technology you like the most?

Code question -> 题做出来 + 沟通

Given two strings s and p, return an array of all the start indices of p's anagrams in s. You may return the answer in any order. 

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Input: s = "cbaebabacd", p = "abc" 
Output: [0,6] 
Explanation: 
The substring with start index = 0 is "cba", which is an anagram of "abc". 
The substring with start index = 6 is "bac", which is an anagram of "abc".

Define own function, write solution

Test case, test the solution step by step
Time complexity

要求：
思路要清晰，做出来同时最佳方法，命名规范
不叫solution




BQ -> 把故事理顺，不要陷入细节，把要点讲清楚
5 projects in your Resume, pick one, if you could, how to make it in a better way? -> 项目了解程度，学习提升的能力，要有一个细节点

Assume you join, and need to learn new things? -> 找资料，带着方案问人
1. Search online for basic understanding, what is about, what can do
2. Go through documentation, by team or start tutorials
3. If still confuse about something ,cannot find solution online or document, ask collegue or manager for help if they are availble. 

What would you do when there's conflict with manager? —> 不要把自己意见藏起来，更加深入研究自己和manager之后再谈
1. make sure explain the opinion clearly to the manager, and understand their 
2. manager knows more than me, and be more , ask for his suggestion
3. Receive manager's feedback, it's good thing to have this talk
4. If he still insist on his, I think I'd better follow him.

Technical difficulty, one example? -> 看能不能坚持、尝试下来

-> Leadership principle

why amazon?
1. 产品好，服务稳定，创新aws平台，提供好的service


LRU再看看

# 08/26/2022

## Coding
### 题目
Given a phone with classical telephone keypad made of numbers from 0 to 9 which also have letters on it (1=abc, 2=def, 3=ghi, 4=jkl, 5=mno, 6=pqr, 7=stu, 8=vwx, 9=yz, 0=space).
Write a program in your favorite programming language that allows to create memos or any free text messages.

This can be useful for users of non-smart-phones or for people with motor or visual disabilities who need help to write faster.

While typing on number keys with the keypad the user sees suggestions of words that have those letters. The suggestions come from a PREDEFINED dictionary.

For example, given a dictionary with the following words: ‘absolute’, ‘amazon’, ‘basket’, ‘bat’, ‘cat’, ‘catalog’, when the user types 117 then the word suggestions will be: ‘bat’ and ‘cat’ 
(considering that key 1 could represent ‘a’ or ‘b’ or ‘c’ and key 7 could represent ‘s’ or ‘t’ or ‘u’).

### 思路
#### 思路一
1. generate all combinations of words using numbers: O
2. find the intersection of the dictionary and combinbation: O(min(M, N))
==> O(3^length(input))

#### 思路二
1. for each word in the dictionary, check if it's a valid suggestion
   1. check the length
   2. iterate through the word, check if can be find through number
   3. when reaches to the end, add it to final result
2. combine all valid suggestions from step 2
==> O(length(word)) * O(size of dict)

#### 思路三
预处理把dictionary转换成数字
117 -> {"cat", "bat"}
==> O(length(word)) * O(size of dict)

### test case
bat, bbo, cat, 117

## 整理回顾
it reminds me to use map, please give me a sec to think about it.
流程要把控好：1. 对于题目要问问题，可以确认一下signature 2. follow-up 自己举test case进一步确认问题 3. 要交流思维过程 4. 要把solution想完备之后再写 5. brute force至少要实现出来，不用非得写出来最优的方法，可以在follow up的时候再提出来，而不用敲

ng：根据OA1, OA2结果
两种：一轮30分钟结束
另一种：三轮每一轮1小时（很可能有一轮OOD面试）去leedcode discuss看most votes板块
BQ：搜leadership principle看面试往里套

内推的话，2-3天就会有回复

非ng有一个team match过程，ng是随机分配


