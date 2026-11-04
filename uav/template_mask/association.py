"""Evaluate an immutable association snapshot before applying any link.

All pair decisions and coordinate-cycle checks see the same template bank.
Rejected proposals cannot alter a later proposal's compatibility decision.
"""
from copy import deepcopy
import numpy as np
from .geometry import discrepancy


def registered_occurrence_owner(engine, supported):
    """Assign duplicate or partial records only with sustained measured layout."""
    from .global_identity import shape_agreement, trajectory_agreement
    e=engine
    ordered=sorted(supported,key=lambda row:row[2].id)
    for n,(ta,fa,a,_) in enumerate(ordered):
        ma=fa.matrix@e.bank.members(ta.id)[a.template_id]
        for tb,fb,b,_ in ordered[n+1:]:
            mb=fb.matrix@e.bank.members(tb.id)[b.template_id]
            relation=np.linalg.inv(ma)@mb
            sigma=fa.uncertainty+fb.uncertainty
            shape=shape_agreement(e,a.template_id,b.template_id,relation,sigma)
            if shape['kind'] not in ('equivalent','a_contains_b','b_contains_a'):return None
            if not trajectory_agreement(e,a,b,relation,sigma,expanding=True)['accepted']:return None
    return ordered[0] if ordered else None


def evaluate_associations(engine):
    # Occurrence grouping is independent of the opposing template. Reuse it
    # within this immutable snapshot only; never across geometry updates.
    engine._association_occurrences = {}
    engine._association_views = {}
    try:
        return _evaluate_associations(engine)
    finally:
        del engine._association_occurrences
        del engine._association_views


def _evaluate_associations(engine):
    from .global_identity import evaluate_pair, shape_agreement
    e = engine
    roots = sorted({min(e.bank.members(t)) for t in e.bank.templates
                    if not e.bank.templates[t].composite})
    proposals = [evaluate_pair(e,a,b) for n,a in enumerate(roots) for b in roots[n+1:]]
    pairs = {(r['a'],r['b']):r for r in proposals}
    neighbours = {t:set() for t in roots}
    for row in proposals:
        if row['accepted']:
            neighbours[row['a']].add(row['b'])
            neighbours[row['b']].add(row['a'])

    def relation(a,b):
        row = pairs[tuple(sorted((a,b)))]
        matrix = np.asarray(row['transform'])
        return (matrix if a<b else np.linalg.inv(matrix)),row['sigma']

    def compatible(a,b,t):
        ab,sab = relation(a,b)
        if t in neighbours[a] and t in neighbours[b]:
            bt,sbt = relation(b,t);at,sat = relation(a,t)
            return discrepancy(ab@bt,at,e.bank.templates[t].box)<=4*(sab+sbt+sat)
        if t in neighbours[b]:
            bt,other_sigma = relation(b,t);implied,target = ab@bt,a
        else:
            at,other_sigma = relation(a,t);implied,target = np.linalg.inv(ab)@at,b
        missing = pairs[tuple(sorted((target,t)))]
        if missing.get('reason') not in ('no_independent_registration','no_supported_global_explanation'):
            return False
        evidence = shape_agreement(e,target,t,implied,sab+other_sigma)
        return evidence['kind'] in ('equivalent','a_contains_b','b_contains_a')

    # Copy only after all checks; a rejection must not change the evidence graph.
    rejected = {(r['a'],r['b']) for r in proposals if r['accepted']
        and any(not compatible(r['a'],r['b'],t)
                for t in (neighbours[r['a']]|neighbours[r['b']])-{r['a'],r['b']})}
    result = deepcopy(proposals)
    for row in result:
        if row['accepted']:
            row['occurrence_groups'] = tuple({t:m.tolist() for t,m in e.bank.members(root).items()}
                                             for root in (row['a'],row['b']))
        if (row['a'],row['b']) in rejected:
            row.update(accepted=False,reason='competing_global_template_groups')
    return result
