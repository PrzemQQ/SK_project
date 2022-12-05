import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt


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
    elif uniform_dist[i] < 0.5:
        free_boxes[1] += 1
    elif uniform_dist[i] < 0.75:
        free_boxes[2] += 1
    else:
        free_boxes[3] += 1

print(free_boxes)
plt.bar(box_kinds, free_boxes) 
plt.show()

print(time_series)

# generowanie rozkładu normalnego ile godzin będzie przechowwyany bagaż



# gnerowanie kolejki klientó rozkład gamma




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
        
    def matrix(self):
        """_summary_

        Returns:
            _type_: matrix with client data (baggage size, hours rented, start time, end time,start hour, end hour)
        """
        return np.array([[self.baggage_size],[self.hours_rented],[self.start_rent],[self.end_rent],[self.starthour_rent],[self.endhour_rent]])
    def __repr__(self):
        return 'Client with baggage size %s, rented for %s hours, from %s to %s' % (self.baggage_size, self.hours_rented, self.start_rent, self.end_rent)
    
    
client1 = Clients('S', 26, start_date, 10)
client2 = Clients('M', 2, start_date, 12)
client3 = Clients('L', 3, start_date, 14)
matrix = np.array([client1.matrix(),client2.matrix(),client3.matrix()])


print(client1)
print(matrix)