The original natural forest dataset, provided to me by Georg, is a uint8 raster.

It seems that openEO does not actually ready the stac metadata for `data_type`, `nodata`, `scale`, `offset`!!
https://github.com/Open-EO/openeo-geotrellis-extensions/issues/658

This causes really tricky bugs downstream.
https://forum.dataspace.copernicus.eu/t/load-stac-configure-nodata/5267

Here we convert the natural forest dataset to a float32 raster, which is hopefully supported by openEO.
This workaround is only practical because we are looking at a small AOI,
and would not be practical to scale up the workflow.
The float32 raster is significantly larger in size.

```bash
gdal_translate -ot Float32 -scale 0 250 0 1 natural-forest/Natural_forest_2020_bugom_32636.tif natural-forest-float32/Natural_forest_2020_bugom_32636.tif
```
