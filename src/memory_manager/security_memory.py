class SecurityMemory:
    def __init__(self):
        self.audits = []
        self.alerts = []

    def store_audit(self, audit):
        self.audits.append(audit)

    def get_audits(self):
        return self.audits

    def register_alert(self, name, payload):
        self.alerts.append({"name": name, "payload": payload})

    def get_alerts(self):
        return list(self.alerts)
