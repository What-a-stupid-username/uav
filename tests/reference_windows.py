"""Reference selection and retention checks; no media access."""
from uav.temporal_blend import window_ranges
from uav.propainter_references import ReferenceBank
bank=ReferenceBank(24,count=6)
for overlap,block in [(5,20),(4,22),(3,24)]:
    ranges=window_ranges(block,10,overlap)
    assert len(ranges)==3
    assert all(b-a==10 for a,b in ranges)
    selections=[]
    for a,b in ranges:
        ids=bank.select(30+a,30+b,150)
        assert len(ids)==6 and len(set(ids))==6
        assert all(not 30+a<=i<30+b for i in ids)
        assert ids==bank.select(30+a,30+b,150)
        selections.append(ids)
    assert selections[0]!=selections[1]
    # Match rolling retention to the next windows' actual reference requests.
    step=block-overlap;keep=step
    needed=set()
    for position in range(keep,min(150,keep+bank.radius+1),step):
        for a,b in window_ranges(min(block,150-position),10,overlap):
            needed.update(i for i in bank.select(position+a,position+b,150) if i<keep)
    bank.frames={i:None for i in range(150)};bank.masks=dict(bank.frames)
    bank.prune(keep,needed)
    for position in range(keep,min(150,keep+bank.radius+1),step):
        for a,b in window_ranges(min(block,150-position),10,overlap):
            assert all(i in bank.frames for i in bank.select(position+a,position+b,150))
assert window_ranges(6)==[(0,6)]
assert window_ranges(15)==[(0,10),(5,15)]
print('independent window references and retention: PASS')
