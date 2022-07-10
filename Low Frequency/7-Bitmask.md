In computers, data are stored as bits. A bit stores either 0 or 1. 
A binary number is a number expressed in the base-2 system. Each digit can be 0 or 1.

010101 = 1\*2^0 + 1\*2^2 + 1\*2^4
in python: 
`bin(21)` returns `010101`

Bit-wise AND:
compare each digit, if both are 1, then resulting digit is 1.

Bitmask
construct a binary number, such that it can turns off all digits except the 1 digit in the mask.
-> 只保留自己是1的位置，把其他位置的1都掩盖掉了

Common operation
`1 << i` access the ith bit in the mask
`bitmask | (1 << i)` set the ith bit in the bitmask to 1
`bitmask & (1 << i)` check if the ith bit in the bitmask is set 1 or not
`bitmask & ~(1 << i)` set the ith bit in the bitmask to 0

pseudocode for DP
```shell
function f(int bitmask, int [] dp) {
    if calculated bitmask {
        return dp[bitmask];
    }
    for each state you want to keep track of {
        if current state not in mask {
            temp = new bitmask;
            dp[bitmask] = max(dp[bitmask], f(temp,dp) + transition_cost);
        }
    }
    return dp[bitmask];
}
```
Bitmask is helpful with problems that would normally require factorial complexity (something like n!) but can instead reduce the computational complexity to 2^n by storing the dp state. 

