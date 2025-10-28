class ForgettingAgent:
    def __init__(self, memory_manager, llm_judge):
        self.memory = memory_manager
        self.judge = llm_judge

    def run(self):
        for event in self.memory.episodic.get_events():
            if self._should_forget(event):
                self.memory.episodic.delete_event(event)
            else:
                verdict = self.judge(f"Classify success/failure:\n{event}")
                if "failure" in verdict.lower():
                    strategy = self._distill_strategy(event)
                    self.memory.semantic.store_concept(f"lesson:{event['id']}", strategy)

    def _should_forget(self, event) -> bool:
        return False

    def _distill_strategy(self, event):
        return self.judge(f"Distill improvement strategy from failure:\n{event}")
