matrix = []
row = [0, 0]
for i in range(2):
    matrix.append(row)
matrix[1][1] = 1
print matrix

-> 
[[0, 1],
 [0, 1]]
因为直接修改了row的值

def modify(elems):
    elems.append("foo")
    elems = ["bar", "baz"]

array = ["qux"]
modify(array)
print array
在def里面赋值并没有影响外面的值，只是会指向一个新的对象


def factory():
    values = []
    def widget(value):
        values.append(value)
        return values
    return widget

worker = factory()
worker(1)
worker(2)
print worker(4)
-> [1, 2, 4]

b = bytearray([0xd9, 0x83, 0xd9, 0x84, 0xd8, 0xa8])
message = b.decode('utf-8')
-> It takes a sequence of bytes and interprets them as UTF-8 encoded Unicode to produce a string of characters.

To make common text smaller, UTF-8 was created. It is a way to encode Unicode characters, that uses a variable-width encoding scheme so that common English characters take less memory. 

In modern operating systems, processes are independent instances of program execution and are made up of one or more threads. Processes are largely independent of each other and only interact through inter-process communication channels. In comparison, threads in the same process share almost all resources, including memory space and file descriptors. As a result, threads are generally faster to create and destroy and can communicate between each other more quickly than processes can. Because threads execute concurrently, each thread needs to maintain independent call stack with information about the current state of execution. The call stack holds local variables, the return address, and the enclosing context of the current operation. But because they share memory with other threads in the same process, threads must synchronize access to shared data to maintain data integrity.

def login(request, cursor):
    cursor.execute("SELECT * FROM users WHERE email = '" \
        + request.params["email"] + "' AND password = '" \
        + request.params["password"] + "'")

-> SQL injection from malicious user input

If an attacker submitted a request with the email parameter set to "admin@example.com" and the password set to "password' OR '1'='1", the database would execute the following query:

SELECT * FROM users WHERE email = 'admin@example.com'
    AND password = 'password' OR '1'='1'
This would bypass the password verification and let the attacker login as the admin. 


    for i in range(len(user_submitted_hash)):
        if user_submitted_hash[i] != database_hash[i]:
          return False
          
-> Timing attacks on the string comparison
The key observation is that the loop that compares the two strings will loop more times (and take longer) if the two hashes share a common prefix. This allows an attacker to go through the hash from beginning to end, and at each character position try all 256 possible bytes. When they guess the correct byte, the comparison will take slightly longer. They can then go on to the next character position, and in short order they'll have extracted the correct hash.