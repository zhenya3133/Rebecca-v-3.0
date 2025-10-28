class ProceduralMemory:
    def __init__(self):
        self.workflows = {}

    def store_workflow(self, name, steps):
        self.workflows[name] = steps

    def get_workflow(self, name):
        return self.workflows.get(name)
