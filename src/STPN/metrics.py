

'''
Include:
corr - with multi variable
occcupancy %

accuracy score

FMMSE

Classification report

confusion matix




'''
# Numpy implementation
import numpy as np

x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])

# Calculate correlation (can NOT have nan!)
c1 = np.corrcoef(x,y)[0,1] # just take the top right value of correlation matrix
# OR
c2 = np.corrcoef(xy)

xyz = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [2, 1, 4, 5, 8, 12, 18, 25, 96, 48],
                [5, 3, 2, 1, 0, -2, -8, -11, -15, -16]])

c3 = np.corrcoef(xyz)


# ========================
# Pandas Implementation: Can handle nan!
import pandas as pd

x = pd.Series(range(10, 20))
y = pd.Series([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
z = pd.Series([5, 3, 2, 1, 0, -2, -8, -11, -15, -16])

x.corr(y, method='pearson') # Pearson's r -- Default (linear relationship)
x.corr(y, method='spearman') # Spearman's rho (monotonic relationship)
x.corr(y, method='kendall') # Kendall's tau


xy = pd.DataFrame({'x-values': x, 'y-values': y})
xy.corr()

xyz = pd.DataFrame({'x-values': x, 'y-values': y, 'z-values': z})
xyz.corr()

