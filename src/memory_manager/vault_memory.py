class VaultMemory:
    def __init__(self):
        self.secrets = {}

    def store_secret(self, name, value):
        self.secrets[name] = value

    def get_secret(self, name):
        return self.secrets.get(name)
