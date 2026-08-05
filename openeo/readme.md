# create conda environment

```bash
cd openeo
conda env create -f environment.yml
```

make sure to activate the environment

```bash
conda activate LEON-P6
```

update the environment

```bash
conda env update -f environment.yml
```

# run unit tests

for the UDFs

```bash
cd openeo 
python -m unittest discover -t . -s test-udf/ -p "*.py"
```
