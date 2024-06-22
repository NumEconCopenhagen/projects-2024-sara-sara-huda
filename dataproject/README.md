# Data analysis project

Our project, titled *Data Project: MONA and VAR-Models*, explores the impact of interest rate changes on economic growth and inflation using the MONA dataset from the Danish Central Bank. We model interactions between key economic variables using a VAR (Vector Autoregression) model and impulse response functions.

We apply the *following datasets*:

1. monadata2023.xlsx (handed to us by bachelor supervisor)
2. cleaned_and_transposed_monadata2023.xlsx

*Methods*
We use data from MONA, focusing on the monetary policy interest rate, output, inflation, and the effective krone exchange rate. We prepare the data, perform stationarity tests, fit a VAR model, and analyze impulse response functions.

*Results*
Our analysis reveals that interest rate shocks initially increase rates and then stabilize. Output and inflation decrease with higher interest rates, while exchange rates show minimal response. Shocks to other variables (exchange rate, output, inflation) show varied impacts, highlighting the dynamic interactions between these economic indicators.

*Conclusion*
Our findings suggest that higher interest rates suppress economic activity and inflation, with little effect on exchange rates. These insights are crucial for the Danish Central Bank's monetary policy decisions.

*Dependencies*
Apart from a standard Anaconda Python 3 installation, the project requires no further installations.

The *full analysis and results* of the project can be seen from running [dataproject.ipynb](dataproject.ipynb).