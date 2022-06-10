import collections
class Product:
    def __init__(self, k):
        self.queue = collections.deque([1])
        self.k = k

    def add(self, n):    
        if n == 0:
            for i in range(len(self.queue)):
                self.queue[i] = 1
            return
        else:
            if len(self.queue) > self.k:
                self.queue.popleft()
            self.queue.append(n * self.queue[-1])

    def get(self):
        if self.k > len(self.queue):
            return self.queue[-1]
        return self.queue[-1] // self.queue[-1 - self.k]

if __name__ == "__main__":
    obj = Product(4)
    obj.add(3)
    obj.add(0)
    obj.add(5)
    obj.add(2)
    obj.add(4)
    print(obj.get())