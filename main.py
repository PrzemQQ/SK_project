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


#start variables
start_date = '2022-01-01'
stop_date = '2022-01-31'
quantity_of_boxes = 100


#parameters
time_series = pd.date_range(start=start_date,end=stop_date, freq='D') #create time series
box_kinds = np.array(['S','M','L','XL']) #box sizes
baggage_kinds = np.array(['S','M','L','XL']) #baggage sizes
uniform_dist = uniform.rvs(size=quantity_of_boxes, loc=0, scale=1) #create uniform distribution
free_boxes = np.array([0,0,0,0]) #free boxes
for i in range(0,len(uniform_dist)): #loop through uniform distribution and assign boxes
    if uniform_dist[i] < 0.25:
        free_boxes[0] += 1
    elif uniform_dist[i] < 0.55:
        free_boxes[1] += 1
    elif uniform_dist[i] < 0.75:
        free_boxes[2] += 1
    else:
        free_boxes[3] += 1
#distribution of free boxes
# print(free_boxes)
# plt.figure()
# plt.title("free boxes")
# plt.bar(box_kinds, free_boxes) 
# plt.show()
# print(time_series)

# generate hours rented
hours_rented = np.random.normal(15,5,100) #generating normal distribution of hours rented
hours_rented_int = [ int(hours_int) for hours_int in hours_rented ] #converting to int

# generate clients per day
clients_per_day = np.random.gamma(15,1.75,100) #generating gamma distribution of clients per day
clients_per_day_int = [ int(clients_int) for clients_int in clients_per_day ] #converting to int


# testing generated data
test_uniform_dist_of_free_boxes = chisquare(free_boxes) #test uniform distribution of free boxes
p_value_uniform_dist_of_free_boxes = test_uniform_dist_of_free_boxes[1]
print(p_value_uniform_dist_of_free_boxes)
plt.figure(figsize = (12, 6)) 
plt.subplot(1,2,1)   
plt.hist(uniform_dist,density = True,bins=4)
plt.axhline(y=uniform.pdf(uniform_dist[0]),color='r')
plt.title('Rozkład jednostajny dla rodzaju bagażu - wersja wygenerowana')
plt.ylabel('Prawdopodobieństwo wystąpienia danego rodzaju bagażu')
plt.xlabel('Rodzaj bagażu')
plt.xticks([0.1,0.3,0.5,0.7],box_kinds)

plt.subplot(1,2,2)
baggage_type = ["S", "M", "L", "XL"]
count = [0.20,0,25,0.30,0.25]
from scipy.stats import uniform

x = uniform.rvs(0.01,0.99,size=1000)     
plt.hist(x,density = True,bins=4)
plt.axhline(y=uniform.pdf(x[0]),color='r')
plt.title('Rozkład jednostajny dla rodzaju bagażu - wersja idealna')
plt.ylabel('Prawdopodobieństwo wystąpienia danego rodzaju bagażu')
plt.xlabel('Rodzaj bagażu')
plt.xticks([0.1,0.3,0.5,0.7],baggage_type)
# plt.show()


test_normal_dist_of_clients = stats.normaltest(clients_per_day_int) #test normal distribution of clients per day
print(test_normal_dist_of_clients)
plt.figure(figsize = (12, 6))
def pdf(x):
    mean = np.mean(x)
    std = np.std(x)
    y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
    return y_out
plt.subplot(1,2,1)
plt.title("Rozkład Normalny  godzin przechowywania bagażu - wersja wygenerowana")
plt.xlabel("Godziny wynajmu")
plt.ylabel("Prawdopodobieństwo wystąpienia danej godziny" )
# plt.hist(clients_per_day_int, density = True, bins = 10)
plt.plot(hours_rented_int, pdf(hours_rented_int), color = 'black')
plt.scatter(hours_rented_int, pdf(hours_rented_int), marker = 'o', s = 25, color = 'red')

plt.subplot(1,2,2) # subplot ideal distribution
x = np.arange(0,30,1)
y = pdf(x)
plt.style.use('seaborn')
plt.title("Rozkład Gaussa godzin przechowywania bagażu - wersja idealna")
plt.xlabel("Godziny wynajmu")
plt.ylabel("Prawdopodobieństwo wystąpienia danej godziny" )
plt.plot(x, y, color = 'black',
         linestyle = 'dashed')
plt.scatter( x, y, marker = 'o', s = 25, color = 'red')
# plt.show()


test_gamma_dist_of_clients_per_day = stats.normaltest(clients_per_day_int) #test gamma distribution of clients per day
print(test_gamma_dist_of_clients_per_day)
plt.figure(figsize = (12, 6))
plt.subplot(1,2,1)
plt.title("Rozkład Gamma ilości klientów - wersja wygenerowana")
plt.xlabel("Ilość klientów")
plt.ylabel("Prawdopodobieństwo wystąpienia danej ilości klientów" )
plt.scatter(clients_per_day_int, pdf(clients_per_day_int), marker = 'o', s = 25, color = 'red')

plt.subplot(1,2,2) # subplot ideal distribution
x = np.linspace(0,60,1000)
y = stats.gamma.pdf(x, a=20, scale=1)

plt.title("Rozkład Gamma ilości klientów - wersja idealna")
plt.xlabel("Ilość klientów")
plt.ylabel("Prawdopodobieństwo wystąpienia danej ilości klientów" )
plt.plot(x,y)
# plt.show()




class Clients:
    """_summary_ = 'Client class for queueing system'
    """
    def __init__(self, baggage_size,hours_rented,start_rent,starthour_rent):
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
        self.endhour_rent = starthour_rent+hours_rented
        if self.endhour_rent > 24: #if the client rents the box for more than 24 hours, the end time is the next day
            self.endhour_rent = self.endhour_rent - 24
            self.end_rent = (pd.to_datetime(start_rent) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        elif self.endhour_rent <24: #if the client rents the box for less than 24 hours, the end time is the same day
            self.end_rent = start_rent
        else:
            self.end_rent = start_rent
        
    def matrix(self):
        """_summary_

        Returns:
            _type_: matrix with client data (baggage size, hours rented, start time, end time,start hour, end hour)
        """
        return np.array([[self.baggage_size],[self.hours_rented],[self.start_rent],[self.end_rent],[self.starthour_rent],[self.endhour_rent]])
    def __repr__(self):
        return 'Client with baggage size %s, rented for %s hours, from %s to %s' % (self.baggage_size, self.hours_rented, self.start_rent, self.end_rent)
    
#testing class clients  
# client1 = Clients('S', 26, start_date, 10)
# client2 = Clients('M', 2, start_date, 12)
# client3 = Clients('L', 3, start_date, 14)
# matrix = np.array([client1.matrix(),client2.matrix(),client3.matrix()])
# print(client1)
# print(matrix)

clients = [] #list of clients
def generate_clients(clients_per_day_int,hours_rented_int):
    """_summary_

    Args:
        clients_per_day_int (_type_): clients per day
        hours_rented_int (_type_): hours rented

    Returns:
        _type_: list of clients
    """
    
    for i in range(0,len(clients_per_day_int)):
        for j in range(0,clients_per_day_int[i]):
            # print(np.random.choice(baggage_kinds),np.random.choice(hours_rented_int),str((np.random.choice(time_series.strftime('%Y-%m-%d')))),np.random.randint(0,24))
            clients.append(Clients(np.random.choice(baggage_kinds),np.random.choice(hours_rented_int),str((np.random.choice(time_series.strftime('%Y-%m-%d')))),np.random.randint(0,24)))
            
    return clients

generate_clients(clients_per_day_int,hours_rented_int)
matrix_clients = np.array([client.matrix() for client in clients]) #matrix with clients data ready to be used in the simulation
print(matrix_clients)