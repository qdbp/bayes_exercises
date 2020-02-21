from bisect import insort_left
from collections import deque
from typing import Deque, List

import numpy as np
import numpy.random as npr

N_MINUTES = 60 * 7


def get_arrival_times():

    arrs = np.cumsum(npr.exponential(scale=10, size=100))
    arrs = arrs[: np.argmax(arrs > N_MINUTES)]

    return arrs


def appoint_arrs(arrs, have_doctors=3):

    waits: List[float] = []
    waitlist: Deque[float] = deque([])

    free_doctors = have_doctors
    events = deque([(t, "arrive") for t in sorted(arrs)])

    def make_appt():
        print("making appointment")
        insort_left(events, (t + npr.uniform(16, 30), "leave"))

    t = 0
    while events:
        t, ev = events.popleft()
        if ev == "arrive":
            print(f"patient arrived at {t=:.1f}", end=";")
            if free_doctors > 0:
                free_doctors -= 1
                assert free_doctors >= 0
                make_appt()
                waits.append(0)
            else:
                print("putting on waitlist")
                waitlist.append(t)
        elif ev == "leave":
            print(f"appointment over at {t=:.1f} ({free_doctors=})")
            if len(waitlist) > 0:
                print(f"getting patient from waitlist of {len(waitlist)}")
                wait_start = waitlist.popleft()
                assert t >= wait_start
                waits.append(t - wait_start)
                make_appt()
            else:
                free_doctors += 1
                assert free_doctors <= have_doctors
        else:
            assert 0, "bad event"

    return waits


def main():

    import matplotlib.pyplot as plt

    arrs = get_arrival_times()

    waits = []
    for _ in range(1000):
        waits.extend(appoint_arrs(arrs))

    plt.hist(waits)
    plt.show()


if __name__ == "__main__":
    main()
