// ------------------------------
// EXPORT OF ESRI LULC 2020
// ------------------------------

// Your region (Bugoma)
var bugoma = ee.Geometry.Rectangle([
  30.5503711040000994,
   1.0709279050000799,
  31.2229521229999989,
   1.5469373050000299
]);

// Load ESRI Global Land Cover (10 m)
var lulc = ee.ImageCollection('projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS')
              .filterDate('2020-01-01', '2020-12-31')
              .mosaic()
              .clip(bugoma);

// Export to Google Drive
Export.image.toDrive({
  image: lulc,
  description: 'ESRI_LULC_2020_Bugoma',
  folder: 'GEE_Exports',          
  fileNamePrefix: 'LULC_2020_Bugoma',
  scale: 10,
  region: bugoma,
  maxPixels: 1e13
});

// Show on Map (optional)
Map.centerObject(bugoma, 10);
Map.addLayer(lulc, {}, 'LULC 2020');

