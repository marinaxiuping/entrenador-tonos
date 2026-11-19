import pytest

from tonos.rate_limit import SlidingWindowRateLimiter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


def test_rejects_requests_over_the_limit_within_the_window(clock):
    limiter = SlidingWindowRateLimiter(max_requests=2, window_s=60, clock=clock)

    assert [limiter.allow("a") for _ in range(3)] == [True, True, False]


def test_allows_again_once_the_oldest_request_leaves_the_window(clock):
    limiter = SlidingWindowRateLimiter(max_requests=2, window_s=60, clock=clock)
    limiter.allow("a")
    clock.now = 30
    limiter.allow("a")

    clock.now = 59
    assert not limiter.allow("a")
    clock.now = 60
    assert limiter.allow("a")
    assert not limiter.allow("a")


def test_counts_each_client_separately(clock):
    limiter = SlidingWindowRateLimiter(max_requests=1, window_s=60, clock=clock)

    assert limiter.allow("a")
    assert limiter.allow("b")
    assert not limiter.allow("a")


def test_rejected_requests_do_not_extend_the_wait(clock):
    limiter = SlidingWindowRateLimiter(max_requests=1, window_s=60, clock=clock)
    limiter.allow("a")
    clock.now = 50
    limiter.allow("a")

    clock.now = 60
    assert limiter.allow("a")


def test_forgets_idle_clients(clock):
    limiter = SlidingWindowRateLimiter(max_requests=1, window_s=60, clock=clock)
    limiter.allow("a")
    clock.now = 120
    limiter.allow("b")

    assert list(limiter._requests) == ["b"]


@pytest.mark.parametrize("max_requests, window_s", [(0, 60), (1, 0)])
def test_rejects_invalid_configuration(max_requests, window_s):
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(max_requests, window_s)
