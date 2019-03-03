import numpy as np
import matplotlib.pyplot as mp
values = [26, 17, 21, 29, 11]
labels = ['Apple', 'Mango', 'Pineapple', 'Banana', 'Strawberry']
mp.figure().set_facecolor(np.ones(3) * 240 / 255)
mp.title('Pie', fontsize=20)
mp.pie(values, (0.05, 0.01, 0.01, 0.01, 0.01), labels,
       ['dodgerblue', 'orangered', 'limegreen', 'violet', 'gold'],
       '%d%%', shadow=True, startangle=90)
mp.axis('equal')
mp.show()
