import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("AnnualMotorVehiclePopulationbyVehicleType.csv")
year = df['year'].tolist()
category = df['category'].tolist()
vehtype = df['type'].tolist()
number = df['number'].tolist()

val1 = df.loc[df['type'] == 'Omnibuses'].index
val2 = df.loc[df['type'] == 'Excursion buses'].index
val3 = df.loc[df['type'] == 'Private buses'].index

#print(val1)
List1 = df.loc[val1]
#print(List1)
#print(val2)
List2 = df.loc[val2]
#print(List2)
#print(val3)
List3 = df.loc[val3]
#print(List3)

plt.plot(List1['year'], List1['number'], label = 'Number of Omnibuses')
plt.plot(List2['year'], List2['number'], label = 'Number of Excursion buses')
plt.plot(List3['year'], List3['number'], label = 'Number of Private buses')
plt.xlabel('Year')
plt.ylabel('Number of vehicles')
plt.xticks(List1['year'])
plt.title('Number of vehicles over the years')
plt.legend()
plt.show()
