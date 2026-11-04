"""Read-only ownership evidence reused within one geometry snapshot."""


class OwnershipEvidence:
    def __init__(self, engine):
        self.engine = engine
        self.owners = tuple(engine.active_instances())
        self.relations = {}
        self.groups = {}
        self.fusions = {}

    def separate(self, a, b):
        """One measured relative-motion decision per pair in this snapshot."""
        from .interactions import relative_motion
        pair = (a.id,b.id)
        if pair not in self.relations:
            self.relations[pair] = relative_motion(self.engine,a,b,a.birth)['state']
        return self.relations[pair] == 'independent_motion'

    def fusion(self, template_id, *, refined):
        key = (template_id,refined,self.engine.bank.revision)
        if key not in self.fusions:
            self.fusions[key] = self.engine.bank.fuse(template_id,coarse=not refined)
        return self.fusions[key]

    def independent(self, members):
        key = tuple(sorted(members))
        if key not in self.groups:
            targets = [i for i in self.owners if i.template_id in members]
            self.groups[key] = tuple(b for b in self.owners
                if b.template_id not in members and any(self.separate(a,b) for a in targets))
        return self.groups[key]
