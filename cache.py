import numpy as np
import pandas as pd
import datetime


class Cache():
    """_summary_ = Cache class for queueing system"""

    def __init__(self, size):
        """_summary_
        Args:
            size (_type_): cache size
            is_free (_type_): boolean indication status of the cache
            occupation_history (_type_): history of cache usage
        """
        self.size = size
        self.is_free = True
        self.end_hour = 0
        self.end_day = pd.to_datetime("2022-01-01").date()
        self.occupation_history = []

    def occupy(self, current_day, current_hour, end_day, end_hour):
        """_summary_ = Occupies cache for a certain number of hours
        Args:
            current_day (_type_): day in which we want to put luggage in the cache
            current_hour (_type_): hour in which we want to put luggage in the cache
            end_day (_type_): day in which we want to put luggage out of the cache
            end_hour (_type_): hour in which we want to put luggage out of the cache
        """
        if self.is_free or self.end_day < pd.to_datetime(current_day).date() or (
                self.end_day == pd.to_datetime(current_day).date() and
                self.end_hour < current_hour):
            self.occupation_history.append([current_day, current_hour, end_day, end_hour])
            self.is_free = False
            self.end_hour = end_hour
            self.end_day = pd.to_datetime(end_day).date()
            return "OK"
        else:
            return "Already occupied"


cache1 = Cache("S")
print(cache1.occupy("2022-01-01", 12, "2022-01-02", 13))
# if cache1.occupy("2022-01-03", 12, "2022-01-03", 13) == "OK":
#     print(1)

print(cache1.end_hour, cache1.end_day)
print(cache1.occupation_history)
