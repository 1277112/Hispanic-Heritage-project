# Hispanic Heritage Historical Data Explorer & Calculus Lab

A self-contained Streamlit classroom app that retrieves World Bank historical indicators, validates/edit observations, fits centered-time polynomial models, calculates estimates, and explores context functions and calculus.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt` (and optionally this README and `sources.md`).
3. In Streamlit Community Cloud, create an app from the repository and select `app.py` as the main file.
4. Deploy. The app retrieves public World Bank data at runtime; no API key is required.

## Deploy on Render

Create a **Python Web Service** connected to the GitHub repository.

Build command:

```bash
pip install -r requirements.txt
```

Start command (single line):

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Verification checklist

- Confirm the displayed country, indicator code, definition, units, source URL, retrieval date, actual year coverage, and observation count.
- Edit a year/value, verify the user-modified scenario label, then choose **Reset to source**.
- Try duplicate years, non-integer years, blank/non-numeric values, negative values for nonnegative indicators, and unemployment above 100%.
- Try polynomial degrees 3–8 with both sufficient and insufficient distinct years.
- Confirm the equation uses centered time, and check R², MAE, residuals, observation count, historical range, and mean.
- Compare single-year and batch estimates for the same year; they should agree.
- Test malformed batch input, duplicate years, years outside 1950–2100, and reversed range endpoints.
- Change country, indicator, degree, sample increment, forward horizon, comparison country, course mode, context, function family, function expression, analysis interval, graph visibility, and current point.
- Test `log(x)`, `(x**2+1)/(x-2)`, `sin(x)`, and invalid expressions. Check undefined points and graph gaps.
- Download CSV, XLSX, and PDF; inspect workbook sheets and PDF chart/metadata/results.
- Deploy and inspect Streamlit/Render logs if startup or API retrieval fails.

## Troubleshooting

- **World Bank timeout / empty response:** retry later; API data may be unavailable or missing for the requested country/indicator.
- **Polynomial degree error:** reduce degree or use more distinct years.
- **Invalid editable table:** correct duplicate/non-integer years and nonnumeric values, then refit.
- **Function parser error:** use `x`, numeric constants, `+ - * / **`, parentheses, and `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `pi`, or `E`.
- **Export error:** check that all dependencies installed successfully; Render and Streamlit Cloud logs show package or runtime exceptions.
- **Deployment port error on Render:** use the exact `$PORT` start command above.

## Important limitations

The polynomial is a descriptive fit, not a causal explanation. Extrapolations can be implausible. R² and MAE are in-sample; chronological holdout MAE is separate. No validated 95% prediction intervals are generated. Numerical root/sign analysis may be approximate. The contextual visualizations are schematic and automatic timed playback is not included in this deployment-safe version.
