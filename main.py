import plotly as py
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import chisquare
from scipy import stats
import statsmodels.api as sm
import pylab as py
from random import randrange

# start variables
start_date = '2022-01-01'
stop_date = '2022-01-31'
quantity_of_boxes = 100

# parameters
time_series = pd.date_range(start=start_date, end=stop_date, freq='D')  # create time series
box_kinds = np.array(['S', 'M', 'L', 'XL'])  # box sizes
baggage_kinds = np.array(['S', 'M', 'L', 'XL'])  # baggage sizes
uniform_dist = uniform.rvs(size=quantity_of_boxes, loc=0, scale=1)  # create uniform distribution
free_boxes = np.array([0, 0, 0, 0])  # free boxes
for i in range(0, len(uniform_dist)):  # loop through uniform distribution and assign boxes
    if uniform_dist[i] < 0.25:
        free_boxes[0] += 1
    elif uniform_dist[i] < 0.55:
        free_boxes[1] += 1
    elif uniform_dist[i] < 0.75:
        free_boxes[2] += 1
    else:
        free_boxes[3] += 1
# distribution of free boxes
# print(free_boxes)
# plt.figure()
# plt.title("free boxes")
# plt.bar(box_kinds, free_boxes) 
# plt.show()
# print(time_series)

# generate hours rented
hours_rented = np.random.normal(15, 5, 100)  # generating normal distribution of hours rented
hours_rented_int = [int(hours_int) for hours_int in hours_rented]  # converting to int

# generate clients per day
clients_per_day = np.random.gamma(15, 1.75, 100)  # generating gamma distribution of clients per day
clients_per_day_int = [int(clients_int) for clients_int in clients_per_day]  # converting to int

# testing generated data
test_uniform_dist_of_free_boxes = chisquare(free_boxes)  # test uniform distribution of free boxes
p_value_uniform_dist_of_free_boxes = test_uniform_dist_of_free_boxes[1]
print(p_value_uniform_dist_of_free_boxes)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(uniform_dist, density=True, bins=4)
plt.axhline(y=uniform.pdf(uniform_dist[0]), color='r')
plt.title('Rozkład jednostajny dla rodzaju bagażu - wersja wygenerowana')
plt.ylabel('Prawdopodobieństwo wystąpienia danego rodzaju bagażu')
plt.xlabel('Rodzaj bagażu')
plt.xticks([0.1, 0.3, 0.5, 0.7], box_kinds)

plt.subplot(1, 2, 2)
baggage_type = ["S", "M", "L", "XL"]
count = [0.20, 0, 25, 0.30, 0.25]
from scipy.stats import uniform

x = uniform.rvs(0.01, 0.99, size=1000)
plt.hist(x, density=True, bins=4)
plt.axhline(y=uniform.pdf(x[0]), color='r')
plt.title('Rozkład jednostajny dla rodzaju bagażu - wersja idealna')
plt.ylabel('Prawdopodobieństwo wystąpienia danego rodzaju bagażu')
plt.xlabel('Rodzaj bagażu')
plt.xticks([0.1, 0.3, 0.5, 0.7], baggage_type)
# plt.show()


test_normal_dist_of_clients = stats.normaltest(clients_per_day_int)  # test normal distribution of clients per day
print(test_normal_dist_of_clients)
plt.figure(figsize=(12, 6))


def pdf(x):
    mean = np.mean(x)
    std = np.std(x)
    y_out = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
    return y_out


plt.subplot(1, 2, 1)
plt.title("Rozkład Normalny  godzin przechowywania bagażu - wersja wygenerowana")
plt.xlabel("Godziny wynajmu")
plt.ylabel("Prawdopodobieństwo wystąpienia danej godziny")
# plt.hist(clients_per_day_int, density = True, bins = 10)
plt.plot(hours_rented_int, pdf(hours_rented_int), color='black')
plt.scatter(hours_rented_int, pdf(hours_rented_int), marker='o', s=25, color='red')

plt.subplot(1, 2, 2)  # subplot ideal distribution
x = np.arange(0, 30, 1)
y = pdf(x)
plt.style.use('seaborn')
plt.title("Rozkład Gaussa godzin przechowywania bagażu - wersja idealna")
plt.xlabel("Godziny wynajmu")
plt.ylabel("Prawdopodobieństwo wystąpienia danej godziny")
plt.plot(x, y, color='black',
         linestyle='dashed')
plt.scatter(x, y, marker='o', s=25, color='red')
# plt.show()


test_gamma_dist_of_clients_per_day = stats.normaltest(clients_per_day_int)  # test gamma distribution of clients per day
print(test_gamma_dist_of_clients_per_day)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Rozkład Gamma ilości klientów - wersja wygenerowana")
plt.xlabel("Ilość klientów")
plt.ylabel("Prawdopodobieństwo wystąpienia danej ilości klientów")
plt.scatter(clients_per_day_int, pdf(clients_per_day_int), marker='o', s=25, color='red')

plt.subplot(1, 2, 2)  # subplot ideal distribution
x = np.linspace(0, 60, 1000)
y = stats.gamma.pdf(x, a=20, scale=1)

plt.title("Rozkład Gamma ilości klientów - wersja idealna")
plt.xlabel("Ilość klientów")
plt.ylabel("Prawdopodobieństwo wystąpienia danej ilości klientów")
plt.plot(x, y)


# plt.show()


class Clients:
    """_summary_ = 'Client class for queueing system'
    """

    def __init__(self, baggage_size, hours_rented, start_rent, starthour_rent):
        """_summary_

        Args:
            baggage_size (_type_): baggage size
            hours_rented (_type_): how many hours the client rented the box
            start_rent (_type_): start time of renting
            end_rent (_type_): end time of renting
            start_hour_rent (_type_): start hour of renting
            end_hour_rent (_type_): end hour of renting
        """

        self.baggage_size = baggage_size
        self.hours_rented = hours_rented
        self.start_rent = start_rent
        self.starthour_rent = starthour_rent
        self.endhour_rent = starthour_rent + hours_rented
        if self.endhour_rent > 24:  # if the client rents the box for more than 24 hours, the end time is the next day
            self.endhour_rent = self.endhour_rent - 24
            self.end_rent = (pd.to_datetime(start_rent) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        elif self.endhour_rent < 24:  # if the client rents the box for less than 24 hours, the end time is the same day
            self.end_rent = start_rent
        else:
            self.end_rent = start_rent

    def matrix(self):
        """_summary_

        Returns:
            _type_: matrix with client data (baggage size, hours rented, start time, end time,start hour, end hour)
        """
        return np.array(
            [[self.baggage_size], [self.hours_rented], [self.start_rent], [self.end_rent], [self.starthour_rent],
             [self.endhour_rent]])

    def __repr__(self):
        return 'Client with baggage size %s, rented for %s hours, from %s to %s' % (
            self.baggage_size, self.hours_rented, self.start_rent, self.end_rent)


# testing class clients
# client1 = Clients('S', 26, start_date, 10)
# client2 = Clients('M', 2, start_date, 12)
# client3 = Clients('L', 3, start_date, 14)
# matrix = np.array([client1.matrix(),client2.matrix(),client3.matrix()])
# print(client1)
# print(matrix)

clients = []  # list of clients


def generate_clients(clients_per_day_int, hours_rented_int):
    """_summary_

    Args:
        clients_per_day_int (_type_): clients per day
        hours_rented_int (_type_): hours rented

    Returns:
        _type_: list of clients
    """

    for i in range(0, len(clients_per_day_int)):
        for j in range(0, clients_per_day_int[i]):
            # print(np.random.choice(baggage_kinds),np.random.choice(hours_rented_int),str((np.random.choice(time_series.strftime('%Y-%m-%d')))),np.random.randint(0,24))
            clients.append(Clients(np.random.choice(baggage_kinds), np.random.choice(hours_rented_int),
                                   str((np.random.choice(time_series.strftime('%Y-%m-%d')))), np.random.randint(0, 24)))

    return clients


generate_clients(clients_per_day_int, hours_rented_int)
matrix_clients = np.array(
    [client.matrix() for client in clients])  # matrix with clients data ready to be used in the simulation
print(matrix_clients)


# baggage class
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

    def info(self):
        """_summary_ = Returns info about the cache
        Returns:
            _type_ = string with info about the cache
        """
        return "Cache size: %s, is free: %s, end hour: %s, end day: %s" % (
            self.size, self.is_free, self.end_hour, self.end_day)

    def info_detal(self):
        """_summary_ = Returns info about the cache
        Returns:
            _type_ = string with info about the cache
        """
        return (self.size, self.is_free, self.end_hour, self.end_day, self.occupation_history)


cache1 = Cache("S")


# print(cache1.occupy("2022-01-01", 12, "2022-01-02", 13))
# print(cache1.occupy("2022-01-03", 12, "2022-01-03", 13))
# print(cache1.end_hour, cache1.end_day)
# print(cache1.occupation_history)


# generate available caches

def generate_caches(cache_sizes, quantity_of_boxes):
    """_summary_

    Args:
        cache_sizes (_type_): list of cache sizes
        quantity_of_boxes (_type_): quantity of all boxes

    Returns:
        _type_: list of caches
    """
    caches = []
    for i in range(0, (
            int(quantity_of_boxes / 4))):  # quantity of boxes divided by 4 because we have 4 sizes of boxes and we want to have the same quantity of each size
        for size in cache_sizes:
            caches.append(Cache(size))
    return caches


generated_caches = generate_caches(box_kinds, quantity_of_boxes)  # list of caches ready to simulate
# print(generated_caches)
print([cache.info_detal() for cache in generated_caches])

# data ready to simulate
matrix_clients
generated_caches

# może ci będzie pomocne do tworzenia symulacji, stworzyłem  osobne mizenne zebys nie musial bawic się w macierz, masz tez to dla ułatweinia w df, wtedy jest łatwiej manipulować danymi,
clients_baggage_size = (matrix_clients[:, 0]).flatten()
clients_hours_rented = matrix_clients[:, 1].flatten()
clients_start_time = matrix_clients[:, 2].flatten()
clients_end_time = matrix_clients[:, 3].flatten()
clients_start_hour = matrix_clients[:, 4].flatten()
clients_end_hour = matrix_clients[:, 5].flatten()
df_clients = pd.DataFrame(
    {'baggage_size': clients_baggage_size, 'hours_rented': clients_hours_rented, 'start_time': clients_start_time,
     'end_time': clients_end_time, 'start_hour': clients_start_hour, 'end_hour': clients_end_hour})
df_clients = df_clients.astype(
    {'baggage_size': 'category', 'hours_rented': 'int64', 'start_time': 'datetime64[ns]', 'end_time': 'datetime64[ns]',
     'start_hour': 'int64', 'end_hour': 'int64'})
df_only_numeric = df_clients.select_dtypes(include=['int64', 'float64'])

df_clients = df_clients.sort_values(by=['start_time', 'start_hour', 'end_time', 'end_hour'])

print(df_clients.head(20))
print(df_clients.describe())
print(generated_caches[75].info_detal())
print(df_clients.iloc[0])

caches_cost = [20, 40, 50, 60]
cache_fine = 60


def knapsack_problem(cache_info, clients_info, caches_costs):
    """_summary_ = returns calculated profit with knapsack method
            Args:
                cache_info (_type_): list of generated caches
                clients_info (_type_): dataframe with clients' info
                caches_costs (_type_): list of cost for renting cache sort by size
            """

    profit = 0

    for i in range(len(clients_info)):
        size = clients_info.iloc[i]['baggage_size']
        start_time = clients_info.iloc[i]['start_time']
        end_time = clients_info.iloc[i]['end_time']
        start_hour = clients_info.iloc[i]['start_hour']
        end_hour = clients_info.iloc[i]['end_hour']
        hours = clients_info.iloc[i]['hours_rented']

        # Adding fine to profit if cache is rented for more than 24 hours
        if hours > 24:
            profit += cache_fine

        # Occupying cache based on luggage size and adding cost of that to profit
        match size:
            case 'XL':
                for cache_number in range(75, 100):
                    if cache_info[cache_number].occupy(start_time, start_hour, end_time, end_hour) == 'OK':
                        profit += caches_costs[3]
                        break
            case 'L':
                for cache_number in range(50, 100):
                    if cache_info[cache_number].occupy(start_time, start_hour, end_time, end_hour) == 'OK':
                        if cache_number >= 75:
                            profit += caches_costs[3]
                        else:
                            profit += caches_costs[2]
                        break
            case 'M':
                for cache_number in range(25, 100):
                    if cache_info[cache_number].occupy(start_time, start_hour, end_time, end_hour) == 'OK':
                        if cache_number >= 75:
                            profit += caches_costs[3]
                        elif cache_number >= 50:
                            profit += caches_costs[2]
                        else:
                            profit += caches_costs[1]
                        break
            case 'S':
                for cache_number in range(0, 100):
                    if cache_info[cache_number].occupy(start_time, start_hour, end_time, end_hour) == 'OK':
                        if cache_number >= 75:
                            profit += caches_costs[3]
                        elif cache_number >= 50:
                            profit += caches_costs[2]
                        elif cache_number >= 25:
                            profit += caches_costs[1]
                        else:
                            profit += caches_costs[0]
                        break
    return profit


def random_profit(cache_info, clients_info, caches_costs):
    """_summary_ = returns calculated profit with random method
            Args:
                cache_info (_type_): list of generated caches
                clients_info (_type_): dataframe with clients' info
                caches_costs (_type_): list of cost for renting cache sort by size
            """

    profit = 0

    for i in range(len(clients_info)):
        size = clients_info.iloc[i]['baggage_size']
        start_time = clients_info.iloc[i]['start_time']
        end_time = clients_info.iloc[i]['end_time']
        start_hour = clients_info.iloc[i]['start_hour']
        end_hour = clients_info.iloc[i]['end_hour']
        hours = clients_info.iloc[i]['hours_rented']

        # Adding fine to profit if cache is rented for more than 24 hours
        if hours > 24:
            profit += cache_fine

        # Occupying cache based on luggage size and adding cost of that to profit
        match size:
            case 'XL':
                for cache_number in range(randrange(75, 100)):
                    if cache_info[cache_number].occupy(start_time, start_hour, end_time, end_hour) == 'OK':
                        profit += caches_costs[3]
                        break
            case 'L':
                for cache_number in range(randrange(50, 100)):
                    if cache_info[cache_number].occupy(start_time, start_hour, end_time, end_hour) == 'OK':
                        if cache_number >= 75:
                            profit += caches_costs[3]
                        else:
                            profit += caches_costs[2]
                        break
            case 'M':
                for cache_number in range(randrange(25, 100)):
                    if cache_info[cache_number].occupy(start_time, start_hour, end_time, end_hour) == 'OK':
                        if cache_number >= 75:
                            profit += caches_costs[3]
                        elif cache_number >= 50:
                            profit += caches_costs[2]
                        else:
                            profit += caches_costs[1]
                        break
            case 'S':
                for cache_number in range(randrange(0, 100)):
                    if cache_info[cache_number].occupy(start_time, start_hour, end_time, end_hour) == 'OK':
                        if cache_number >= 75:
                            profit += caches_costs[3]
                        elif cache_number >= 50:
                            profit += caches_costs[2]
                        elif cache_number >= 25:
                            profit += caches_costs[1]
                        else:
                            profit += caches_costs[0]
                        break
    return profit


print(knapsack_problem(generated_caches, df_clients, caches_cost))
print(random_profit(generated_caches, df_clients, caches_cost))
