// ============================================
// Natural Forest Extraction - Uganda
// ============================================

// Define Uganda region
var uganda = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
  .filter(ee.Filter.eq('country_na', 'Uganda'))
  .geometry();

// Load the dataset
var probabilities = ee.ImageCollection(
  'projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection')
  .mosaic()
  .select('B0');

// Visualise
Map.centerObject(uganda, 7);
Map.addLayer(
  probabilities.mask(probabilities.neq(0)),
  {min: 0, max: 250, palette: ['white', 'green']},
  'Natural forest probabilities'
);

// ============================================
// EXPORT UGANDA 
// ============================================

Export.image.toDrive({
  image: probabilities.clip(uganda),
  description: 'natural_forest_uganda_2020',
  folder: 'GEE_Exports',
  fileNamePrefix: 'natural_forest_uganda',
  scale: 10,
  region: uganda,
  maxPixels: 1e13,
  crs: 'EPSG:32636'   // UTM Zone 36N (Uganda)
});

print('✓ Export task for Uganda created!');
print('Go to Tasks tab (top right) and click RUN.');