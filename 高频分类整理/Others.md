[271. Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/)
组合大文字的时候用数字+特殊字符来连接，decode时候就需要找到数字大小

时间：O(N), N is num of word in words
空间：O(1)

```py
class Codec:
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        res = ""
        for s in strs:
            res += str(len(s)) + "#" + s
        return res
        

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        res = []
        i = 0
        
        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1
            length = int(s[i : j])
            res.append(s[j + 1: j + 1 + length])
            i = j + 1 + length
        
        return res
        
# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))
```