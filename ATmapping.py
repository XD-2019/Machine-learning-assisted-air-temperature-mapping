import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from pykrige.rk import RegressionKriging
from sklearn.linear_model import LinearRegression

met = pd.read_csv("sample_data/AWS_test_sample.csv").set_index('index')
dem = pd.read_csv("sample_data/elevation.csv", index_col=0)

lons, lats = dem.columns.astype(float), dem.index.astype(float)
xgrid, ygrid = np.meshgrid(lons, lats)
lons1d, lats1d = xgrid.flatten(), ygrid.flatten()

indexdf = pd.DataFrame(np.array(range(len(lons1d))).reshape(300, 300), index=lats, columns=lons)

v_dict = pd.DataFrame({'Lon': lons1d, 'Lat': lats1d,
                       'elevation': dem.values.flatten(),
                       'BD': pd.read_csv("sample_data/BD.csv", index_col=0).values.flatten(),
                       'Low_vege': pd.read_csv("sample_data/Low_vege.csv", index_col=0).values.flatten(),
                       'ISF': pd.read_csv("sample_data/ISF.csv", index_col=0).values.flatten(),
                       'Road': pd.read_csv("sample_data/Road.csv", index_col=0).values.flatten(),
                       'Water': pd.read_csv("sample_data/Water.csv", index_col=0).values.flatten(),
                       })

"""
Prepare input data
"""
variables = ['elevation', 'BD', 'Low_vege', 'ISF', 'Road', 'Water']
input_df = met.copy()

for var in variables:
    for key in input_df.index:
        lon, lat = input_df.loc[key].longitude, input_df.loc[key].latitude
        cloni, clati = np.argmin(abs(lons - lon)), np.argmin(abs(lats - lat))
        if lon > lons.max() or lon < lons.min() or lat > lats.max() or lat < lats.min():
            # print(key)
            continue
        data = v_dict[var][indexdf.iloc[clati, cloni]]
        input_df.loc[key, var] = data

input_df = input_df.dropna()

x = input_df[['longitude', 'latitude']].to_numpy()
p = input_df[variables].to_numpy()
y = input_df['air temperature'].to_numpy()

"""
Select your regression model
"""

# m = RandomForestRegressor(n_estimators=175)
m = LinearRegression(fit_intercept=True)
# m = MLPRegressor(hidden_layer_sizes=(500),
#                         random_state=5, max_iter=2000,
#                         learning_rate_init=0.001,
#                         activation='logistic')

m_rk = RegressionKriging(regression_model=m, variogram_model='power', coordinates_type='geographic')
m_rk.fit(p, x, y)

x_t = np.array([lons1d, lats1d]).T
p_t = v_dict[variables].to_numpy()

rg_res_grid = m.predict(p_t)
rk_res_grid = m_rk.predict(p_t, x_t)

"""
visualize results
"""

res = rk_res_grid.reshape(len(lats), len(lons))

tick_max = None
nbins, minum_n = 50, 50
if tick_max == None:
    tick_max, tick_min = np.nanpercentile(rk_res_grid, 99), np.nanpercentile(rk_res_grid, 1)

levels = matplotlib.ticker.MaxNLocator(nbins=nbins, min_n_ticks=minum_n).tick_values(tick_min, tick_max)
cmap = cm.get_cmap('RdYlBu', 128)
newcolors = np.vstack((cmap(np.linspace(0, 1, 128)[::-1])))
norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
cmap = matplotlib.colors.ListedColormap(newcolors, name='BuYlRd')

pcm = plt.pcolormesh(xgrid, ygrid, res, cmap=cmap, norm=norm)

plt.colorbar(pcm)
