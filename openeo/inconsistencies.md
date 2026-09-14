# Inconsistencies

Differences between this openEO implementation,
and the reference xarray implementation, documented in the ATBD.

## Tree cover density

This openEO implementation excludes `CLMS_TCD_PANTROPICAL_10M_YEARLY_V1`
values above 100 (unclassifiable, no_data).

## Linear speckle filtering

This openEO implementation applies Sentinel-1 speckle filters in linear scale, rather than in dB.

## Native resolution speckle filtering

This openEO implementation applies speckle filters on native resolution data.
Specifically, the coefficient of variation of the noise applied here is `1 / sqrt(4)`,
which is appropriately `1 / sqrt(L)` where `L` is the equivalent number of looks.

The reference xarray implementation applies the same equivalent number of looks but to lower resolution (resampled) data.

## Speckle filter edge handling

The reference implementation uses scipy `mode="reflect"` / dask `boundary="reflect"`.
Equivalent functionality is not available in openEO.

## Time range

This openEO implementation pads the time range by 3 months at each end.
To allow for full temporal windows centered on 2020-01-01 and 2024-12-31.

## Logistic fitting NaNs

This openEO implementation does not skip NaNs in the logistic fit.

The reference xarray implementation does skip NaNs in sum squared error,
which allows for windows which contain NaNs to appear to have a low sum squared error.

## Logistic fitting vectorised

The reference implementation fits candidate pixels in serial.
This openEO implementation processes all pixels in parallel.

## Assigning effective date of deforestation based on connected area

For pixels in which no deforestation event was detected,
but which are considered to be deforested due to a connected area constraint.

This openEO implementation fills the effective "date of deforestation"
with the date of a nearest deforested pixel.

The reference xarray implementation loops over years, and fills the effective "date of deforestation" as YYYY-01-01.

## KPI units

This openEO implementation delivers all KPIs in hectares.
This is a hard constraint based on the capabilites of processing vector data in openEO.
https://forum.dataspace.copernicus.eu/t/merge-vector-cubes/5425
