// ------------------------------
// EXPORT OF ESA WorldCover 2020
// ------------------------------

// Your region (Bugoma)
var bugoma = ee.Geometry.Rectangle([
  30.5503711040000994,
   1.0709279050000799,
  31.2229521229999989,
   1.5469373050000299
]);

// Load ESA WorldCover 2020 (10 m)
var worldcover = ee.Image('ESA/WorldCover/v100').clip(bugoma);

// Export to Google Drive
Export.image.toDrive({
  image: worldcover,
  description: 'ESA_WorldCover_2020_Bugoma',
  folder: 'GEE_Exports',          
  fileNamePrefix: 'ESA_WorldCover_2020_Bugoma',
  scale: 10,
  region: bugoma,
  maxPixels: 1e13
});

// Show on Map (optional)
Map.centerObject(bugoma, 10);
Map.addLayer(worldcover, {}, 'ESA WorldCover 2020');