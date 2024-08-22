import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np

# Define the UTM projection (Yoko projection)
utm_crs = "EPSG:32633"  # UTM zone 33N, WGS84

# Import Camera positions
gps_cameras = pd.read_excel("GPS Caméras.xlsx")
gps_cameras['geometry'] = gps_cameras.apply(lambda row: Point(row['X'], row['Y']), axis=1)
xy = gpd.GeoDataFrame(gps_cameras, geometry='geometry', crs="EPSG:4326")  # Assuming initial data is in WGS84

# Transform the CRS to match the Canopy height data
canopyheigh = gpd.read_file("Canopyheigh.shp")  # Read Canopy height data to get its CRS
xy = xy.to_crs(canopyheigh.crs)

# Save the shapefile
output_dir = "wrd"
xy.to_file(f"{output_dir}/XY.shp")

# ---------------------- River -----------------------
river = gpd.read_file("/vsizip/./River.zip/River.shp")

# Transform River CRS to match Canopy height data
river = river.to_crs(canopyheigh.crs)

# Cast to LINESTRING if necessary
if river.geom_type[0] != 'LineString':
    river = river.explode().reset_index(drop=True)
    river['geometry'] = river['geometry'].apply(lambda geom: geom[0] if isinstance(geom, list) else geom)
    river = river.set_geometry('geometry').explode().reset_index(drop=True)

# Calculate the distance between xy points and the nearest river
dist = xy.geometry.apply(lambda geom: river.distance(geom).min())

# Create DataFrame for minimum distance
mindist_moddata = pd.DataFrame({'X': xy.geometry.x, 'Y': xy.geometry.y, 'Near_DistRiv': dist})

# Save the minimum distance to an Excel file
mindist_moddata.to_excel("Mindist_ModData.xlsx", index=False)

# ----------------------- Sumriver -----------------------
buffer_sizes = [250, 500, 750, 1000, 2000]

# Initialize a results matrix
result_matrix = np.zeros((len(xy), len(buffer_sizes)))

# Calculate sum of river lengths within buffers
for i, point in xy.iterrows():
    for j, buffer_size in enumerate(buffer_sizes):
        buffer = point.geometry.buffer(buffer_size)
        clipped_river = river.intersection(buffer)
        sum_length = clipped_river.length.sum()
        result_matrix[i, j] = sum_length

# Combine the results with the coordinates
cam_sumriverlength = pd.DataFrame(result_matrix, columns=[f"CamRivbuf_{s}" for s in buffer_sizes])
cam_sumriverlength.insert(0, "X", xy.geometry.x)
cam_sumriverlength.insert(1, "Y", xy.geometry.y)

# Save the results to a text file
cam_sumriverlength.to_csv("AllBuff_CamLocSumRiverLenght.txt", sep='\t', index=False)

# Another way: Calculate sum of river lengths within each buffer size
result_matrix2 = []
for buffer_size in buffer_sizes:
    buffer = xy.geometry.buffer(buffer_size)
    clipped_river = river.intersection(buffer)
    sum_length = clipped_river.length.sum()
    result_matrix2.append(sum_length)

result_matrix2 = np.array(result_matrix2).T

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Point
from scipy.spatial import distance_matrix
import numpy as np
from rasterio.plot import show
from rasterstats import zonal_stats
import openpyxl

# Define the UTM projection (Yoko projection)
utm_crs = "EPSG:32633"  # UTM zone 33N, WGS84

# Import Habitat shapefile
hab_path = "/vsizip/./River.zip/Habitats.shp"
hab = gpd.read_file(hab_path)
canopyheigh = gpd.read_file("Canopyheigh.shp")
hab = hab.to_crs(canopyheigh.crs)

# Calculate Euclidean distance between all pairs of camera and house locations
combined = pd.concat([xy, hab], ignore_index=True)
distances = distance_matrix(combined.geometry.apply(lambda geom: (geom.x, geom.y)), 
                            combined.geometry.apply(lambda geom: (geom.x, geom.y)))

# Extract distances from each camera to all house locations
camera_to_house_distances = distances[:len(xy), len(xy):]

# Find the minimum distance for each camera
min_distances = camera_to_house_distances.min(axis=1)

# ------------read border------------------
ycf_path = "/vsizip/./River.zip/YokoBorder.shp"
ycf = gpd.read_file(ycf_path)

# -------------------Plotting--------------------------------
# Plot Canopy height raster and overlay points and boundaries
with rasterio.open("Canopyheight.tif") as canopyheigh:
    show(canopyheigh, title="Canopy Height with Camera Locations")
    xy.plot(ax=show(canopyheigh, title="Canopy Height with Camera Locations"), color='red', markersize=10)

# Extract Canopy Height data for the points
canopyheigh_data = rasterio.open("Canopyheight.tif")
canopyheigh_values = [val[0] for val in canopyheigh_data.sample([(geom.x, geom.y) for geom in xy.geometry])]

# Save Canopy Height data to Excel
canopyheigh_moddata = pd.DataFrame(canopyheigh_values, columns=['CanopyHeight'])
canopyheigh_moddata.to_excel("Canopyheigh_xy.xlsx", index=False)

# Read and plot DEM
dem = rasterio.open("./Elev_Yoko.tif")
show(dem, title="DEM")

# Crop Human Footprint raster to match the DEM extent
hf = rasterio.open("./wildareas-v3-2009-human-footprint.tif")
hf_crs = dem.crs  # Get the CRS from DEM
hf = rasterio.warp.reproject(hf, dst_crs=hf_crs)

# Plot Human Footprint raster
show(hf, title="Human Footprint")
xy = xy.to_crs(hf_crs)

# Calculate zonal statistics for the Human Footprint raster within buffers
buffer_sizes = [250, 500, 750, 1000, 2000]
hf_moddata = []

for buf_size in buffer_sizes:
    buffers = xy.geometry.buffer(buf_size)
    hf_stats = [zonal_stats(buffer, hf, stats=['mean'])[0]['mean'] for buffer in buffers]
    hf_moddata.append(hf_stats)

hf_moddata_df = pd.DataFrame(hf_moddata).T
hf_moddata_df.columns = [f'CamRivbuf_{s}' for s in buffer_sizes]

# Save HF moddata to a text file
hf_moddata_df.to_csv("AllBuff_CamLocSumRiverLenght.txt", sep='\t', index=False)

# Read other rasters (TRI, Slope, TPI)
tri = rasterio.open("./TRI_Yoko.tif")
slope = rasterio.open("./Slope_Yoko.tif")
tpi = rasterio.open("./TPI_Yoko.tif")

# Calculate and save TPI zonal statistics for buffers
tpi_moddata = []

for buf_size in buffer_sizes:
    buffers = xy.geometry.buffer(buf_size)
    tpi_stats = [zonal_stats(buffer, tpi, stats=['mean'])[0]['mean'] for buffer in buffers]
    tpi_moddata.append(tpi_stats)

tpi_moddata_df = pd.DataFrame(tpi_moddata).T
tpi_moddata_df.columns = [f'TPI_{s}' for s in buffer_sizes]

# Combine with camera coordinates and save to Excel
tpi_moddata_df = pd.concat([pd.DataFrame(xy.geometry.apply(lambda geom: (geom.x, geom.y)).tolist(), columns=['X', 'Y']), tpi_moddata_df], axis=1)
tpi_moddata_df.to_excel("TPI.xlsx", index=False)

# If needed, write to text file
tpi_moddata_df.to_csv("GrdiB15.txt", sep='\t', index=False)

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson, NegativeBinomial
from sklearn.preprocessing import scale
import openpyxl

# Set working directory
wrd = os.getcwd()

# Download file
url = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_elev.zip"
file_name = "Elevation.zip"
file_path = os.path.join(wrd, file_name)

response = requests.get(url)
with open(file_path, 'wb') as file:
    file.write(response.content)

# Load required data
file = "FinalDBMassoh.txt"
ModData = pd.read_csv(file, delimiter="\t", decimal=",", quotechar='"')

# Count occurrences of each species
species_counts = ModData['Species'].value_counts().reset_index()
species_counts.columns = ['Species', 'Abundance']

# Merge counts back to the original dataframe
ModData = pd.merge(ModData, species_counts, on='Species', how='left')

# Convert categorical columns to 'category' dtype
ModData['Season'] = ModData['Season'].astype('category')
ModData['Habitat'] = ModData['Habitat'].astype('category')
ModData['Visibility'] = ModData['Visibility'].astype('category')

# Convert numeric columns to 'numeric' dtype
numeric_cols = ["Abundance", "Dem", "Slope", "TPI", "TRI", "Cheigh", 
                "NeaDistRiv", "DistRoad", "Dist_Nath", "Effort"]
ModData[numeric_cols] = ModData[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Reorder levels (if needed)
ModData['Season'] = ModData['Season'].cat.reorder_categories(['PSP'], ordered=True)
ModData['Habitat'] = ModData['Habitat'].cat.reorder_categories(['Savannah'], ordered=True)
ModData['Visibility'] = ModData['Visibility'].cat.reorder_categories(['Close'], ordered=True)

# Correlation matrix
corrM = ModData[['Dem', 'Slope', 'TPI', 'TRI', 'Cheigh', 'NeaDistRiv', 
                 'DistRoad', 'Dist_Nath', 'Effort']].corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corrM, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

# Histogram plots
test_vars = ["Dem", "Slope", "TPI", "TRI", "Cheigh", "NeaDistRiv", 
             "DistRoad", "DistVillage", "Effort", "Dist_Nath"]

fig, axs = plt.subplots(3, 4, figsize=(15, 10))
axs = axs.ravel()

for i, var in enumerate(test_vars):
    axs[i].hist(ModData[var].dropna(), bins=30, color='blue', alpha=0.7)
    axs[i].set_title(var)

plt.tight_layout()
plt.show()

# Scaling predictors
ModData['s.Slope'] = np.sqrt(ModData['Slope'])**2
ModData['s.Dem'] = ModData['Dem']
ModData['s.TPI'] = ModData['TPI']
ModData['s.TRI'] = np.sqrt(np.sqrt(ModData['TRI']))
ModData['s.Cheigh'] = ModData['Cheigh']
ModData['s.NeaDistRiv'] = np.sqrt(np.sqrt(ModData['NeaDistRiv']))
ModData['s.DistRoad'] = np.sqrt(np.sqrt(ModData['DistRoad']))
ModData['s.Dist_Nath'] = np.sqrt(np.sqrt(ModData['Dist_Nath'])**2)
ModData['s.Effort'] = np.log(ModData['Effort'])

# Z-transforming covariates
pred_var = ["s.Slope", "s.Dem", "s.TPI", "s.TRI", "s.Cheigh", 
            "s.NeaDistRiv", "s.DistRoad", "s.Effort", "s.Dist_Nath"]
ModData[pred_var] = scale(ModData[pred_var])

# Refit model with negative binomial
mod_formula = "Abundance ~ Season + Habitat + Visibility + z.s.Slope + z.s.Dem + z.s.TPI + z.s.TRI + z.s.Cheigh + z.s.NeaDistRiv + z.s.DistRoad + z.s.DistVillage + z.s.Effort + z.s.Dist_Nath"
model = glm(mod_formula, data=ModData, family=NegativeBinomial()).fit()
print(model.summary())

# Calculate overdispersion
Chat = model.deviance / model.df_resid
print(f"Overdispersion: {Chat}")

# Save dataframe
ModData.to_excel("Modata.xlsx", index=False)

# Visualization: Boxplot per season
sns.boxplot(x='Abundance', y='Season', data=ModData)
plt.savefig("Seasonabun.png", dpi=1800)
plt.show()

# Visualization: Boxplot per habitat
sns.boxplot(x='Abundance', y='Habitat', data=ModData)
plt.savefig("Hababun.png", dpi=1800)
plt.show()

# Visualization: Boxplot per visibility
sns.boxplot(x='Abundance', y='Visibility', data=ModData)
plt.savefig("VisiAbun.png", dpi=1800)
plt.show()

# Grouped Boxplot: Habitat by Season
P2 = ModData.groupby('Habitat')['Abundance'].mean().reset_index()
sns.boxplot(x='Habitat', y='Abundance', data=ModData, palette="pastel")
for i in range(len(P2)):
    plt.text(i, 0.5, f"M = {P2['Abundance'].iloc[i]:.1f}", ha='center')
    plt.text(i, 0, f"SD = {ModData['Abundance'].std():.1f}", ha='center')
plt.savefig("Hababun_PerSeason.png", dpi=1800)
plt.show()

# Grouped Boxplot: Habitat by Visibility
P3 = ModData.groupby('Visibility')['Abundance'].mean().reset_index()
sns.boxplot(x='Visibility', y='Abundance', data=ModData, palette="pastel")
for i in range(len(P3)):
    plt.text(i, 0.5, f"M = {P3['Abundance'].iloc[i]:.1f}", ha='center')
    plt.text(i, 0, f"SD = {ModData['Abundance'].std():.1f}", ha='center')
plt.savefig("AbperVisi.png", dpi=1800)
plt.show()

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests
import numpy as np
from scipy import stats
import xlsxwriter
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import broom

# Assuming 'modList' is a list of fitted models and 'bestModel' is the best model based on AIC
modList = [model1, model2, model3]  # Replace with your model list
bestModel = modList[0]  # Replace with the index of the best model

# Compare models using AIC
aic_values = [model.aic for model in modList]
best_aic = min(aic_values)

# Save AIC values to an Excel file
df_aic = pd.DataFrame({'Model': ['Model1', 'Model2', 'Model3'], 'AIC': aic_values})
df_aic.to_excel("AIC.xlsx", index=False)

# Summary of the best model
model_summary = bestModel.summary2().tables[1]
model_summary.to_excel("bestModel.xlsx")

# Model output with confidence intervals
model_output = bestModel.get_robustcov_results().summary_frame(alpha=0.05)
model_output.to_excel("model_output.xlsx")

# Test for multicollinearity (VIF)
y, X = dmatrices(bestModel.formula, data=Data, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.to_excel("vif.xlsx", index=False)

# DFFITS and DFBETAs for assessing model stability
dffits_values = bestModel.get_influence().dffits[0]
dfbeta_values = bestModel.get_influence().dfbeta
dfbeta_df = pd.DataFrame(dfbeta_values, columns=X.columns)
dfbeta_df.to_excel("dfbeta_values.xlsx", index=False)

# Model fit statistics
pseudo_r_squared = 1 - (bestModel.deviance / bestModel.null_deviance)
r_squared_glmm = pseudo_r_squared  # Approximated

# Exponentiated coefficients (IRR)
exp_coef = np.exp(bestModel.params)
conf = bestModel.conf_int()
irrs = np.exp(conf)
results = pd.DataFrame({
    'Estimate': bestModel.params,
    'IRR': exp_coef,
    'Lower_95_CI': irrs[0],
    'Upper_95_CI': irrs[1],
    'P_value': bestModel.pvalues
})
results.to_excel("results.xlsx", index=False)

# Plot IRRs
plt.figure(figsize=(10, 6))
sns.barplot(x=results.index, y=results['IRR'], ci=None)
plt.xticks(rotation=90)
plt.savefig("Irr.png", dpi=1800, bbox_inches='tight')

# Plot model effects using ggplot-style plots (with seaborn)
plt.figure(figsize=(10, 6))
sns.lineplot(x='z.s.NeaDistRiv', y='Abundance', data=Data)
plt.savefig("PredRiv.png", dpi=1800, bbox_inches='tight')

# Additional plots as per your requirements (e.g., for different predictors)
# ...

# Model diagnostics
fitted_vals = bestModel.fittedvalues
residuals = bestModel.resid_deviance
plt.figure(figsize=(10, 6))
sns.residplot(x=fitted_vals, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.savefig("residuals_fitted.png", dpi=1800, bbox_inches='tight')

# Save drop1 results
drop1_results = bestModel.get_influence().summary_frame()
drop1_results.to_excel("drop1.xlsx", index=False)

# Save other relevant plots and summaries similarly
# ...

# diversity indices and plot
# Install the necessary packages (uncomment if not already installed)
# !pip install numpy pandas seaborn openpyxl iNextPy scipy

# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy, f_oneway
from iNextPy import iNext, ggiNext

# Load RAC_Habitats data
RAC_Habitats = pd.read_excel("RAC_Habitats.xlsx")

# Convert to a numpy matrix, excluding the first column (assumed to be labels)
data_matrix = RAC_Habitats.iloc[:, 1:].to_numpy()

# Check if there are any missing values in the matrix
print("Any missing values in data_matrix:", np.any(np.isnan(data_matrix)))

# Generate rarefaction curves
rac_Hab = ggiNext(iNext(data_matrix[:, 3:], q=0, datatype="abundance"), type=1, facet_var="Assemblage")
rac_Hab2 = ggiNext(iNext(data_matrix[:, :-2], q=0, datatype="abundance"), type=1, facet_var="Assemblage")
rac_Hab3 = ggiNext(iNext(data_matrix[:, :-2], q=0, datatype="abundance"), type=1, facet_var="Assemblage")

# Save the plot
rac_Hab3.savefig("rac_GSvsSF_Ss.png", dpi=1800)

# Create a dataframe with species data
Rac3 = pd.DataFrame({
    'Species': [0, 127, 1, 395, 34, 1, 19, 0, 44, 1, 2, 45, 2, 20, 9,
                0, 25, 1, 3, 5, 0, 98, 0, 14, 6, 0, 355, 8, 153, 11,
                7, 36, 89, 124, 38, 0, 131, 12, 72, 14, 3, 19, 3, 94, 18,
                0, 4, 0, 0, 0, 0, 3, 1, 21, 2, 0, 27, 1, 17, 5,
                10, 79, 2, 102, 9, 0, 169, 2, 166, 13, 0, 13, 48, 8, 11,
                0, 13, 0, 5, 2, 8, 25, 12, 17, 14, 0, 54, 2, 40, 8,
                0, 1, 0, 0, 0, 2, 60, 0, 102, 5, 0, 14, 0, 0, 0,
                1, 2, 10, 6, 0, 0, 0, 7, 0, 1, 1, 29, 1, 15, 4,
                0, 36, 27, 7, 13, 0, 94, 9, 13, 1, 0, 25, 13, 14, 2,
                0, 2, 0, 2, 2, 0, 0, 0, 1, 1, 2, 24, 0, 14, 6,
                0, 1669, 37, 613, 58, 7, 68, 4, 42, 9, 0, 0, 0, 0, 2,
                2, 16, 1, 25, 2, 0, 5, 2, 21, 3, 0, 5, 5, 5, 4,
                2, 12, 1, 0, 0, 8, 83, 15, 228, 24]
})

# Save to an Excel file
Rac3.to_excel("RacAllspecies.xlsx", index=False)

# Convert the data to a matrix
data_matrix4 = Rac3.to_numpy()

# Generate rarefaction curve for all species
Rac_ALL = ggiNext(iNext(data_matrix4, q=0, datatype="abundance"), type=1, facet_var="Assemblage")
Rac_ALL.savefig("Rac_ALL.png", dpi=1800)

# Diversity indices functions
def shannon_diversity(x):
    return entropy(x, base=np.e)

def simpson_diversity(x):
    return 1 - np.sum((x/np.sum(x))**2)

# Calculate Shannon and Simpson diversity indices
shannon = np.apply_along_axis(shannon_diversity, 0, data_matrix)
simpson = np.apply_along_axis(simpson_diversity, 0, data_matrix)

# Calculate number of species (S)
S = np.sum(data_matrix > 0, axis=0)

# Pielou's evenness index
pielou = shannon / np.log(S)

# Store results in a dataframe
diversity_df = pd.DataFrame({
    'Habitat': RAC_Habitats.columns[1:],  # Assuming first column is habitat names
    'Shannon_Index': shannon,
    'Simpson_Index': simpson,
    'Pielou_Index': pielou
})

# Perform ANOVA on Shannon indices
shannon_aov = f_oneway(*[diversity_df.loc[diversity_df['Habitat'] == h, 'Shannon_Index'] for h in diversity_df['Habitat']])
print("ANOVA results for Shannon diversity: ", shannon_aov)

# Load RAC_Season data and process
RAC_Season = pd.read_excel("RAC_Season.xlsx")
data_matrix3 = RAC_Season.iloc[:, 1:].to_numpy()

# Generate rarefaction curves for seasons
rac_Season1 = ggiNext(iNext(data_matrix3[:, :-2], q=0, datatype="abundance"), type=1, facet_var="Assemblage")
rac_Season2 = ggiNext(iNext(data_matrix3[:, -2:], q=0, datatype="abundance"), type=1, facet_var="Assemblage")

# Save the plot
rac_Season2.savefig("Rac_Season2.png", dpi=1800)

# Calculate mean and standard errors for each habitat
mean_shannon = np.mean(shannon)
se_shannon = np.std(shannon) / np.sqrt(len(shannon))

# Store the results in a dataframe
results_df = pd.DataFrame({
    'Habitat': RAC_Season.columns[1:],  # Adjust as needed
    'Mean_Shannon_Index': mean_shannon,
    'Shannon_Standard_Error': se_shannon,
    'Shannon_Results': f"{mean_shannon:.2f} ± {se_shannon:.2f}"
})

# Print the results
print(results_df)
