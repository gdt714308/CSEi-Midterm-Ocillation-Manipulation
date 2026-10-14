# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cartopy==0.25.0",
#     "gcsfs==2025.10.0",
#     "marimo",
#     "matplotlib==3.10.7",
#     "numpy==2.2.6",
#     "polars==1.35.1",
#     "xarray==2025.10.1",
#     "zarr==3.1.3",
# ]
# ///


__generated_with = "0.25.1"

# %%


# %%
import marimo as mo
import pickle
import zarr
import xarray as xr
xr.set_options(display_style='text')
import polars as pl
import gcsfs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

gcs = gcsfs.GCSFileSystem(token='anon')

# %%
era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xr.open_zarr(era5_path, chunks=None, storage_options=dict(token='anon'))
full_era5

# %%
# Variables for WBGT
good_vars = [
    'total_sky_direct_solar_radiation_at_surface',
    '2m_temperature', 
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind'
]

# Get a small test dataset (2 days in January 1970)
subset = full_era5[good_vars].sel(time=slice('1970-01-01','1970-01-02'))

# European region (60°N to 40°N, 0°E to 40°E)
europe = subset.sel(latitude=slice(60,40), longitude=slice(0,40))

print(f"✓ Europe data loaded: {europe.dims}")

# %%
# Save for later use
europe.to_netcdf('era5_test_for_wbgt.nc')
print("✓ Saved: era5_test_for_wbgt.nc")

# %%
# Explore available variables
full_era5.data_vars

# %%
mo.md(r"""
# WBGT Calculation Steps

Below we'll implement the Wet Bulb Globe Temperature calculation
""")

# %%
mo.md(r"""
## Step 1: Calculate wet bulb temperature
""")

# %%
# Calculate relative humidity from temperature and dewpoint
def calculate_relative_humidity(temp_k, dewpoint_k):
    """Calculate RH using August-Roche-Magnus approximation"""
    T = temp_k - 273.15  # Convert to Celsius
    Td = dewpoint_k - 273.15

    a = 17.625
    b = 243.04

    gamma_T = (a * T) / (b + T)
    gamma_Td = (a * Td) / (b + Td)

    rh = 100 * np.exp(gamma_Td - gamma_T)

# Calculate wet bulb temperature (simplified Stull formula)
def calculate_wet_bulb_temp(temp_k, dewpoint_k):
    """Calculate wet bulb temperature in Kelvin"""
    T = temp_k - 273.15
    Td = dewpoint_k - 273.15

    rh = calculate_relative_humidity(temp_k, dewpoint_k)

    # Stull (2011) formula
    Tw = T * np.arctan(0.151977 * (rh + 8.313659)**0.5) + \
         np.arctan(T + rh) - np.arctan(rh - 1.676331) + \
         0.00391838 * rh**1.5 * np.arctan(0.023101 * rh) - 4.686035


# Calculate WBT for our data
wbt = calculate_wet_bulb_temp(
    europe['2m_temperature'], 
    europe['2m_dewpoint_temperature']
)
wbt.attrs['long_name'] = 'Wet Bulb Temperature'
wbt.attrs['units'] = 'K'

print(f"✓ Wet bulb temperature calculated")
print(f"  Range: {float(wbt.min()):.1f}K to {float(wbt.max()):.1f}K")

# %%
mo.md(r"""
## Step 2: Calculate 2m wind speed
""")

# %%
# Calculate wind speed from components
u_wind = europe['10m_u_component_of_wind']
v_wind = europe['10m_v_component_of_wind']

wind_speed = np.sqrt(u_wind**2 + v_wind**2)
wind_speed.attrs['long_name'] = '10m Wind Speed'
wind_speed.attrs['units'] = 'm/s'

print(f"✓ Wind speed calculated")
print(f"  Range: {float(wind_speed.min()):.1f} to {float(wind_speed.max()):.1f} m/s")

# %%
mo.md(r"""
## Step 3: Calculate globe temperature

Globe temperature accounts for radiant heat from solar radiation
""")

# %%
# Calculate solar zenith angle
def calculate_zenith_angle(latitude, time):
    """Simplified solar zenith angle calculation"""
    day_of_year = time.dt.dayofyear
    hour = time.dt.hour + time.dt.minute / 60.0

    # Solar declination
    declination = 23.45 * np.sin(2 * np.pi * (day_of_year - 81) / 365)

    # Hour angle
    hour_angle = 15 * (hour - 12)

    # Convert to radians
    lat_rad = np.deg2rad(latitude)
    dec_rad = np.deg2rad(declination)
    ha_rad = np.deg2rad(hour_angle)

    # Solar zenith angle
    cos_zenith = (np.sin(lat_rad) * np.sin(dec_rad) + 
                  np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))

    zenith_angle = np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))

# Calculate globe temperature
def calculate_globe_temp(temp_k, solar_rad, wind_spd, zenith):
    """Calculate globe temperature from air temp, solar radiation, and wind"""
    T_celsius = temp_k - 273.15

    # Convert solar radiation (J/m²) to W/m² if needed
    solar_w = solar_rad / 3600.0 if solar_rad.max() > 5000 else solar_rad

    # Effective solar radiation (accounting for sun angle)
    cos_zenith = np.cos(np.deg2rad(zenith))
    cos_zenith = np.where(cos_zenith > 0, cos_zenith, 0)  # Night = 0
    effective_solar = solar_w * cos_zenith

    # Avoid division by zero
    wind_safe = np.where(wind_spd < 0.5, 0.5, wind_spd)

    # Additional heating from solar radiation
    solar_heating = 0.02 * (effective_solar / np.sqrt(wind_safe))

    globe_temp = T_celsius + solar_heating

# Calculate zenith angle and globe temperature
zenith = calculate_zenith_angle(europe['latitude'], europe['time'])

tg = calculate_globe_temp(
    europe['2m_temperature'],
    europe['total_sky_direct_solar_radiation_at_surface'],
    wind_speed,
    zenith
)
tg.attrs['long_name'] = 'Globe Temperature'
tg.attrs['units'] = 'K'

print(f"✓ Globe temperature calculated")
print(f"  Range: {float(tg.min()):.1f}K to {float(tg.max()):.1f}K")

# %%
mo.md(r"""
## Step 4: Calculate WBGT

**Formula:** WBGT = 0.7 × WBT + 0.2 × TG + 0.1 × TAS
""")

# %%
tas = europe['2m_temperature']

# Calculate WBGT
wbgt = 0.7 * wbt + 0.2 * tg + 0.1 * tas
wbgt.attrs['long_name'] = 'Wet Bulb Globe Temperature'
wbgt.attrs['units'] = 'K'

# Convert to Celsius for interpretation
wbgt_celsius = wbgt - 273.15

print(f"✓ WBGT calculated")
print(f"  Range: {float(wbgt_celsius.min()):.1f}°C to {float(wbgt_celsius.max()):.1f}°C")
print(f"  Mean: {float(wbgt_celsius.mean()):.1f}°C")

# Heat stress statistics
high_risk = (wbgt_celsius > 31).sum()
moderate_risk = ((wbgt_celsius > 27) & (wbgt_celsius <= 31)).sum()
total_points = wbgt_celsius.size

print(f"\nHeat Stress Risk Assessment:")
print(f"  High risk (>31°C): {float(high_risk)/total_points*100:.1f}%")
print(f"  Moderate risk (27-31°C): {float(moderate_risk)/total_points*100:.1f}%")

# %%
mo.md(r"""
# Visualizations

Now let's create maps and plots of our data
""")

# %%
# Simple temperature map
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

temp_celsius = europe['2m_temperature'].isel(time=0) - 273.15

temp_celsius.plot(
    ax=ax, 
    transform=ccrs.PlateCarree(),
    cmap='RdBu_r',
    cbar_kwargs={'label': 'Temperature (°C)'}
)

ax.coastlines(linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

ax.set_title('2m Air Temperature - Europe', fontsize=14, fontweight='bold')
plt.tight_layout()
fig

# %%
# WBGT map with heat stress thresholds
fig2 = plt.figure(figsize=(12, 8))
ax2 = plt.axes(projection=ccrs.PlateCarree())

wbgt_plot = wbgt_celsius.isel(time=0)

wbgt_plot.plot(
    ax=ax2,
    transform=ccrs.PlateCarree(),
    cmap='YlOrRd',
    vmin=0, vmax=40,
    cbar_kwargs={'label': 'WBGT (°C)'}
)

# Add threshold contours
CS = ax2.contour(
    wbgt_plot.longitude, wbgt_plot.latitude, wbgt_plot,
    levels=[27, 31],
    colors='black',
    linewidths=2,
    transform=ccrs.PlateCarree()
)
ax2.clabel(CS, inline=True, fontsize=10, fmt='%d°C')

ax2.coastlines(linewidth=1)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

ax2.set_title('Wet Bulb Globe Temperature (WBGT)\nBlack lines: Heat stress thresholds (27°C, 31°C)', 
              fontsize=14, fontweight='bold')
plt.tight_layout()
fig2

# %%
# Wind speed map
fig3 = plt.figure(figsize=(12, 8))
ax3 = plt.axes(projection=ccrs.PlateCarree())

wind_plot = wind_speed.isel(time=40)

wind_plot.plot(
    ax=ax3,
    transform=ccrs.PlateCarree(),
    cmap='viridis',
    cbar_kwargs={'label': 'Wind Speed (m/s)'}
)

ax3.coastlines(linewidth=1)
ax3.add_feature(cfeature.BORDERS, linewidth=0.5)
ax3.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

ax3.set_title('10m Wind Speed', fontsize=14, fontweight='bold')
plt.tight_layout()
fig3

# %%
def _():
    # Wind speed map
    fig5 = plt.figure(figsize=(12, 8))
    ax5 = plt.axes(projection=ccrs.PlateCarree())

    wind_plot = wind_speed.isel(time=0)


    wind_plot.plot(
        ax=ax3,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        cbar_kwargs={'label': 'Wind Speed (m/s)'}
    )

    ax5.coastlines(linewidth=1)
    ax5.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax5.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    ax3.set_title('10m Wind Speed', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig5


_()

# %%
def Trial_PLot():
    fig6 = plt.figure(figsize=(12,8))
    ax6 = plt.axes(projection=ccrs.PlateCarree())

    return fig6

# %%
# Multi-panel comparison
fig4, axes = plt.subplots(2, 2, figsize=(16, 12),
                          subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

# Temperature
temp_plot = (europe['2m_temperature'].isel(time=0) - 273.15)
temp_plot.plot(ax=axes[0], transform=ccrs.PlateCarree(), cmap='RdBu_r',
               cbar_kwargs={'label': '°C', 'shrink': 0.8})
axes[0].set_title('Air Temperature', fontweight='bold')
axes[0].coastlines(linewidth=0.5)

# Dewpoint
dew_plot = (europe['2m_dewpoint_temperature'].isel(time=0) - 273.15)
dew_plot.plot(ax=axes[1], transform=ccrs.PlateCarree(), cmap='RdYlGn',
              cbar_kwargs={'label': '°C', 'shrink': 0.8})
axes[1].set_title('Dewpoint Temperature', fontweight='bold')
axes[1].coastlines(linewidth=0.5)

# Solar radiation
solar_plot = europe['total_sky_direct_solar_radiation_at_surface'].isel(time=0)
solar_plot.plot(ax=axes[2], transform=ccrs.PlateCarree(), cmap='YlOrRd',
                cbar_kwargs={'label': 'J/m²', 'shrink': 0.8})
axes[2].set_title('Solar Radiation', fontweight='bold')
axes[2].coastlines(linewidth=0.5)

# WBGT
wbgt_celsius.isel(time=0).plot(ax=axes[3], transform=ccrs.PlateCarree(), 
                                cmap='YlOrRd',
                                cbar_kwargs={'label': '°C', 'shrink': 0.8})
axes[3].set_title('WBGT (Heat Stress)', fontweight='bold')
axes[3].coastlines(linewidth=0.5)

plt.suptitle('Climate Variables Affecting Heat Stress', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
fig4