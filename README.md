# flow-duration-analyzer
A simple Python script to analyze flow duration data and plot the results.

## Usage
The script takes a CSV file as input and generates a plot of the flow duration curve. The CSV file should have two columns: the first column should contain the flow duration values, and the second column should contain the corresponding flow values.

To run the script, use the following command:
```
python flow_duration_analyzer.py input.csv output.png
```

Where `input.csv` is the input CSV file and `output.png` is the output plot file.

## Example
Suppose you have a CSV file `data.csv` with the following contents:
```
Flow Duration,Flow Value
1,10
2,20
3,30
4,40
5,50
6,60
7,70
8,80
9,90
10,100
```

You can run the script as follows:
```
python flow_duration_analyzer.py data.csv flow_duration_curve.png
```

This will generate a plot `flow_duration_curve.png` with the flow duration curve.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

**Output**:
```python
['README.md']
```

