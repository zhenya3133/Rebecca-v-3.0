class CoreMemory:
    def __init__(self):
        self.data = {}

    def store_fact(self, key, value):
        self.data[key] = value

    def get_fact(self, key):
        return self.data.get(key)
