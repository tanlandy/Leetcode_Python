[535. Encode and Decode TinyURL](https://leetcode.com/problems/encode-and-decode-tinyurl/)
两个字典，分别存{long:short}和{short:long}

```python

class Codec:
    codeDB, urlDB = defaultdict(), defaultdict()
    chars = string.ascii_letters + string.digits

    def getCode(self) -> str:
        code = ''.join(random.choice(self.chars) for i in range(6))
        return "http://tinyurl.com/" + code
 
    def encode(self, longUrl: str) -> str:
        if longUrl in self.urlDB: 
            return self.urlDB[longUrl]

        code = self.getCode()
        while code in self.codeDB:
             code = getCode()
             
        self.codeDB[code] = longUrl
        self.urlDB[longUrl] = code
        return code

    def decode(self, shortUrl: str) -> str:
        return self.codeDB[shortUrl]

```


[49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)

用一个count记录每个string的字母出现次数，然后res的values存最后的结果

时间：O(M*N), M is len(strs), N is average len(one string)
空间：O(M*N)

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # {charCount: oneRes}
        res = collections.defaultdict(list) 
        
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord("a")] += 1
            res[tuple(count)].append(s)
        
        return res.values()
```

[238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

走左右两遍

时间：O(N)
空间：O(1)
```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * (len(nums))
        
        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]
            
        postfix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]
            
        return res
```