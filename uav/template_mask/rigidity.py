"""Contrary motion of disjoint parts of one original observation.

Consumes existing image-pair RANSAC results; does not search images, fit motion
priors, or use predictions. Repeated copies using the same reference points
are different occurrences, not evidence against the reference's rigidity.
"""
import numpy as np
from .geometry import discrepancy


class RigidityEvidence:
    def __init__(self):
        self.pairs = {}

    def observe(self, reference, source, fits, box):
        """Return true only after two originals show a layout change."""
        for n,a in enumerate(fits):
            ia=set(a.reference_indices)
            if not ia:continue
            for b in fits[n+1:]:
                ib=set(b.reference_indices)
                if not ib or len(ia&ib)>.1*min(len(ia),len(ib)):continue
                layout=np.linalg.inv(a.matrix)@b.matrix
                sigma=a.uncertainty+b.uncertainty
                history=self.pairs.setdefault(reference,[])
                for previous in history:
                    if previous['source']==source:continue
                    # Follow the same two source regions, allowing RANSAC to
                    # choose different subsets; pair order has no meaning.
                    def overlap(x,y):return len(x&y)/max(1,min(len(x),len(y)))
                    if min(overlap(ia,previous['a']),overlap(ib,previous['b']))>=.5:
                        old=previous['layout']
                    elif min(overlap(ia,previous['b']),overlap(ib,previous['a']))>=.5:
                        old=np.linalg.inv(previous['layout'])
                    else:continue
                    if discrepancy(layout,old,box)>8*(sigma+previous['sigma']):
                        return True
                # Bounded raw evidence per reference; repeated queries from
                # one original never corroborate themselves.
                if not any(p['source']==source for p in history):
                    history.append(dict(source=source,a=ia,b=ib,layout=layout,sigma=sigma))
                    del history[:-8]
        return False
