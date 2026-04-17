// Download Crop Probability Map from DynamicWorld

// Define date range and region
var START = ee.Date('2024-01-01');
var END = ee.Date('2025-12-31');
var REGION = ee.Geometry.Point([31.0, 1.0]).buffer(50000); // 50 km buffer for example

// Filter Dynamic World collection
var dwCol = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(REGION)
  .filterDate(START, END)
  .select('crops'); // only agriculture probability

// Compute mean probability over the period
var meanCrops = dwCol.mean().rename('meancrops_24-25');

// Visualise mean crop probability
Map.centerObject(REGION, 8);
Map.addLayer(meanCrops, {min: 0, max: 1, palette: ['white', 'green']}, 'Mean Crop Probability');

// ---- Export ONLY the meanCrops image ----
Export.image.toDrive({
  image: meanCrops,
  description: 'MeanCrops_2024_2025',
  fileNamePrefix: 'meancrops_24-25',
  region: REGION,           
  scale: 10,               
  maxPixels: 1e13,         
  fileFormat: 'GeoTIFF'
});