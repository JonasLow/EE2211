import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("government-expenditure-on-education.csv")
expenditureList = df['recurrent_expenditure_total'].tolist()
yearList = df['year'].tolist()

plt.plot(yearList, expenditureList, label = 'Expenditure over the years')
plt.xlabel('Year')
plt.ylabel('Expenditure')
plt.title('Education Expenditure')
plt.show()
