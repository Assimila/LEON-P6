# create conda environment

```bash
cd openeo
conda env create -f environment.yml
```

# run unit tests

for the UDFs

```bash
cd openeo 
python -m unittest discover -t . -s test-udf/ -p "*.py"
```
