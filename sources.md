# Sources

## Historical indicators

All historical observations are requested from the World Bank Indicators API:

https://api.worldbank.org/v2/country/{ISO3}/indicator/{INDICATOR_CODE}?date={START}:{END}&format=json

Countries offered: Mexico (MEX), Brazil (BRA), Argentina (ARG), Colombia (COL), Chile (CHL), Peru (PER), Venezuela (VEN), Ecuador (ECU), and Uruguay (URY).

| Indicator | World Bank code |
|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` |
| GDP per capita | `NY.GDP.PCAP.CD` |
| Crude birth rate | `SP.DYN.CBRT.IN` |
| Net migration | `SM.POP.NETM` |
| Intentional homicide rate | `VC.IHR.PSRC.P5` |
| Consumer price inflation | `FP.CPI.TOTL.ZG` |
| Central government debt | `GC.DOD.TOTL.GD.ZS` |
| Total reserves including gold | `FI.RES.TOTL.CD` |
| External balance on goods/services | `NE.RSB.GNFS.ZS` |

The application requests current year minus 70 through current year minus 1. It displays actual non-null coverage and count because the API may not provide every year.

## API documentation

- World Bank Indicators API: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation
- World Bank DataBank / indicators: https://data.worldbank.org/indicator

## Mathematical and software references

- NumPy polynomial fitting: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
- SymPy calculus and symbolic differentiation: https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html
- Streamlit documentation: https://docs.streamlit.io/
- ReportLab user guide: https://docs.reportlab.com/
