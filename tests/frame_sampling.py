"""Timestamp mapping tests require no media access."""
from uav.frame_sampling import sample_count, source_index


def main():
    assert sample_count(22120, 60, 24, 15) == 360
    assert sample_count(900, 60, 24) == 360
    assert [source_index(i, 60, 24) for i in range(6)] == [0, 2, 5, 7, 10, 12]
    assert [source_index(i, 12, 24) for i in range(6)] == [0, 0, 1, 1, 2, 2]
    for count, rate, target in [(22120,60,24),(101,29.97,24),(9,12,24),(8,24,24)]:
        n=sample_count(count,rate,target)
        indices=[source_index(i,rate,target) for i in range(n)]
        assert indices == sorted(indices)
        assert indices[0] == 0 and indices[-1] < count
        assert 0 <= n/target-count/rate < 1/target+1e-9
    for bad in [0, -1, float('inf'), float('nan')]:
        try:sample_count(100,60,bad)
        except ValueError:pass
        else:raise AssertionError('invalid FPS accepted')
    print('Frame sampling: 60/29.97/24/12 FPS, duration and invalid rates passed')


if __name__ == '__main__':main()
