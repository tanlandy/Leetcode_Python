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

## Feedback

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

# 09/22/2022

Learn and Be Curious
tell me about a time when you didn't know what to do next or how to solve a challenging problem. How do you learn what you don't know? what were the options you considered? How did you decide the best path forward? What was the outcome?

Bias for Action / Deliver Results
Give me a time when you were able to deliver an important project under a tight deadline. What sacridices did you have to make to meet the deadline? What was thefinal outcome?

- finished before, but after that was told to give a presentation
- sacrafice between meeting deadline and your work

## coding

implement a file searching library:

1. find all files that have a given size requirement: eg. find all files over 5 MB under a directory
2. find all files that match a specified string pattern: eg. find all files with "Grocery" in the name

File file
   String getPath()
   int getSize()
   List<File> children()
   boolena isDir()

leadership principle

```java
// I would like you to implement a file searching library. That can do the following:

//     Find all files that have a given size requirement, eg "Find all files over 5 MB somewhere under a directory"
//     Find all files that match a specified string pattern eg. "Find all files with Grocery in the name"

// Keep in mind that these are just 2 uses cases and that the library should be flexible. Asking a new question, such as finding all pdf files modified less then 2 days ago, shouldn't be too difficult.


File file
  String getPath() -> /tmp/var/log/
  int getSize() -> 1024
  Date getLastModifiedTime() -> 2022/01/01 10:24:10.123
  List<File> children() -> [File(/tmp/var/log/1.log, /tmp/var/log/2.log, ...)]
  boolean isDir();
  
  
public Solu {
   
      public Lis<File> BFS(File root, Rule rule) {
        Queue<File> queue = new LinkedList<>();
          List<File> res = new LinkedList<>();
          queue.offer(root);
          while(!queue.isEmpty()) {
            File curr = queue.poll();
              if(!curr.isDir()) {
               res.add(curr);
               
                continue;
              }
             List<File> childs = curr.children();
              for(File f:childs) {
               queue.add(f);
              }
          }
          return res;
      }
      
      
      public void main() {
        File root = new File("/home/xxx")
          Rule rule = new OversizeRule(1024.0);
          List<File> res = BFS(root, newrule);
      }
}

public Rule {
  public boolean compare(File f, String str) {
      return ture;
    }
}

public OverSize extend Rule {
  public boolean compare(File f, String weight) {
      return f.getSize() > Integer.valueOf(weight);
    }
}
public Smaller extend OverSize {
  public boolean compare(File f, String larger, String smaller) {
      if(super.compare(f, weight)) {
          return f.getsize() < smaller;
        }
        return false;
    }
}

public Contains extend Rule {

 public boolean compare(File f, String str) {
      // String.contains(string);
    return f.getPath().contains(str);
  }
}

 > 5 MB && .contains("hello")

2 MB < size < 5 MB

public interface Rule {
   boolean compare(File f);
}

public OverSizeRule implements Rule {
 private double threshold;
  
  public OverSizeRule(double threshold) {
   this.threshold = threshold;
  }
  
  public boolean compare(File f) {
   return f.getSize() > this.threshold;
  }
}

public AndRule implements Rule {
 List<Rule> rules;
  
  public boolean compare(File f) {
   for(Rule r: this.rules) {
          if(!r.compare(f)) {
           return false;
          }
    }
    return true;
  }
}

```

# 10/03/22

introduction

## BQ

tell me a time that you find you cannot meet your commitment. What was that, how did you risk, and how did you communicate
--> STAR + Learnt
timeline!

tell me time when you need to understand some complicated problem? --> 看怎么dive deep，如何debug，而不是直接放弃，怎么意识到是complicated然后如何去做的
--> clarify why that is not the best approach
--> why do you use the other one

--> STAR + Learnt
what is the most challenging part

--> follow STAR to answer
ownership answer should focus on when you relized you cannot meet the commitment, how do you communicate to other people and how to do tradeoffs to make the right decisions
you should proactively move things forward by discussing with managers / peers, doing research to dive deep, etc.

## Coding via codeshare

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input

对于复制来的题，也要自己给面试官讲一遍题目确保自己了解了题目。问clarifying questions: positive number? sorted?

对于4轮

1. LP + coding (data structure and algo)
2. LP + coding (logical and maintainable) -> break into helper function
3. LP + coding (problem solving) -> edge cases: "Mary spends $5.5 this week. Bob spends $10" -> discount amount by 20%: $1,000, %5.5有小数点，有大数的 -> follow up是其他钱
4. LP only

# 11/5/23 - internship

1. ownership
tell me a time when you need to help your teammate. why did you help, what did you do, and what's the result?
加进了一个新人
how much time you spent on your own tasks, and how much time you spent for the teammate
take lead on project: 发现问题，开会解决问题
willingness to take tasks not your task
demonstrate impact

2.

a time you need to make a quick decision when you don't have time
bias

```py
"""

When customers place orders on Amazon, 
they can choose to deliver the packages to lockers located at Amazon pick up locations. 
Packages can come in many different sizes. We have lockers of different sizes as well. 
The bigger size locker can fit smaller/same size package, but not the other way. 
Can you implement an algorithm to efficiently find the best possible empty locker for a given package? 
You are given 3 helper classes to begin with. Feel free to change the class, add members, etc. as necessary.

"""

class Package(packageId):
    String packageId;
    int size;
    
class Locker:
    String LockerId;
    String packageId;
    int size;

class PickupLocation:
    String pickupLocation;
    lockers;

# a list of sorted lockers by size


def fit(package, lockers):
    locker = ''
    for i, c in enumerate(lockers):
        if c.size > package.size and c.packageId == '': 
            locker = c.LockerId
            c.packageId = package.packageId
            break
    if locker == '': return "Cannot fit package"
    
    return locker

def pickup(Lockers, lId):
    pId = ''
    for i in Lockers:
        if i.LockerId == lId: 
            pId = i.packageId
            i.packageId = ''
            return pId
    
    return "package not found"
            

def newFit(package, lockersUnUsed, lockersUsed):
    if lockersUnUsed[package.size]:
        lockersUnUsed[package.size][0].packgeId = package.packageId
        id = lockersUnUsed[package.size][0].lockerId
        lockerUsed[package.size].append(lockersUnUsed.pop(0))
        return id
    else: return ''

```
