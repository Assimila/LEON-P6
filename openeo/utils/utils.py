import sys
import pprint

import openeo
import openeo.processes


def describe_collection(
    connection: openeo.Connection,
    collection_id: str,
    stream=sys.stdout,
):
    """
    Prints information about the collection

    The summaries key tells you what you can filter on using properties argument of load_collection.
    https://open-eo.github.io/openeo-python-client/api.html#openeo.rest.connection.Connection.load_collection
    """
    desc = connection.describe_collection(collection_id)
    for key, value in desc.items():
        print(f"{key}:", file=stream)
        pprint.pprint(value, stream=stream)


def percentile_cube(
    cube: openeo.DataCube,
    aoi,
    percentile,
) -> openeo.DataCube:
    """
    Calculate `percentile` over spatial `aoi`.

    The `aoi` argument is required because the openEO process `reduce_spatial` is not available 🙁

    Arguments:
        cube: The input data cube.
            Should have dimensions of a least (x, y).
        aoi: A geojson Parameter.
            Should contain a single polygon geometry over the area of interest.
            Pixels outside the aoi will be masked out.
        percentile: A number Parameter.
            Should have value between 0 and 1.

    Returns:
        A DataCube like `cube` but where the pixel values are equal to the calculated percentile.
    """

    def calculate_percentile(data: openeo.processes.ProcessBuilder):
        return data.quantiles(probabilities=[percentile])

    # reduce_spatial not available
    percentile_vectorcube = cube.aggregate_spatial(
        geometries=aoi,
        reducer=calculate_percentile,
        # target_dimension argument not supported?!
        # target_dimension = "result",
    )

    # Experimental openEO process
    # based on https://github.com/Open-EO/openeo-community-examples/blob/49033e3709be1d709bd5b0a918e94c1833964f6e/python/RankComposites/utils_BAP.py#L25
    percentile_datacube = percentile_vectorcube.vector_to_raster(cube)

    return percentile_datacube


def convert_to_dB(cube: openeo.DataCube) -> openeo.DataCube:

    def calculate_dB(data: openeo.processes.ProcessBuilder):
        return 10.0 * data.log(10)
    
    return cube.apply(calculate_dB)


def drop_hidden_dimension(cube: openeo.DataCube, name: str) -> openeo.DataCube:
    """
    Workaround for https://forum.dataspace.copernicus.eu/t/load-stac-time-dimension-hidden/5312
    """
    try:
        cube.drop_dimension(name)
    except ValueError as e:
        if e.args[0].startswith("No dimension named"):
            return cube.process(
                process_id="drop_dimension",
                arguments={"data": cube, "name": name},
                metadata=cube.metadata or None,
            )
        else:
            raise
