"""
Hispanic Heritage Historical Data Explorer & Calculus Lab
Self-contained Streamlit application.
"""
from __future__ import annotations
import io, math, re, time
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

st.set_page_config(page_title="Hispanic Heritage Data Lab", page_icon="📊", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn is expected to live under current mortality patterns.", "Higher values indicate longer life expectancy; this is a population-level estimate."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, national estimate.", "This is a rate, not a count of unemployed people."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "Gross domestic product divided by midyear population, in current US dollars.", "Not average wealth, household income, or a measure of how income is distributed."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual number of live births per 1,000 people.", "A rate, not the total number of births."),
    "Net migration": ("SM.POP.NETM", "people (estimated net number)", "Net number of migrants during the period: immigrants minus emigrants.", "Not an emigration count; positive and negative values represent net gain or loss."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Victims of intentional homicide per 100,000 people.", "A rate; definitions and reporting may differ across countries and years."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index.", "Inflation rate, not the price level."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP.", "Coverage is central government, not necessarily all public-sector debt."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, valued in current US dollars.", "Nominal dollar value; not adjusted for inflation."),
    "External balance on goods/services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP.", "A positive value is a surplus; a negative value is a deficit."),
}
CONTEXTS = {
    "Business": ("Revenue", "US dollars", "Time", "months", "A business indicator can describe revenue or output over time. Its derivative is the rate of change in revenue per month."),
    "Engineering": ("Position", "meters", "Time", "seconds", "For motion, the function is position; its first derivative is velocity and its second derivative is acceleration."),
    "Medical": ("Concentration", "mg/L", "Time", "hours", "A concentration model can represent how a measured substance changes over time; derivatives describe its rate and acceleration of change."),
    "Sports science": ("Distance", "meters", "Time", "seconds", "A movement model can describe distance or position; derivatives can represent speed/velocity and acceleration."),
    "Biology": ("Population", "organisms", "Time", "days", "A population model describes the number of organisms; derivatives represent population growth rate and changing growth rate."),
    "Bioengineering": ("Cell count", "cells", "Time", "hours", "A cell-growth model describes cell count; derivatives represent growth rate and change in growth rate."),
    "Economic forecasting": ("Economic index", "index points", "Time", "years", "An economic model describes an index over time; derivatives represent the pace and acceleration of change."),
    "Weather": ("Temperature", "°C", "Time", "hours", "A temperature model describes temperature over time; derivatives represent warming/cooling rate and how that rate changes."),
    "Custom": ("Output", "units", "Input", "units", "Describe what the variables mean in your own context."),
}
SUGGESTIONS = {
    "Business": "a*x**2 + b*x + c", "Engineering": "a*x**2 + b*x + c",
    "Medical": "A*exp(-k*x) + C", "Sports science": "a*sin(b*x + c) + d",
    "Biology": "K/(1 + A*exp(-r*x))", "Bioengineering": "K/(1 + A*exp(-r*x))",
    "Economic forecasting": "a*x**2 + b*x + c", "Weather": "A*sin(w*x + p) + C",
    "Custom": "a*x + b",
}
WB_ROOT = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

def fetch_world_bank(country: str, indicator: str, start: int, end: int):
    url = WB_ROOT.format(country=country, indicator=indicator)
    params = {"date": f"{start}:{end}", "format": "json", "per_page": 2000}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            return pd.DataFrame(columns=["year", "value"]), f"No observations returned by the World Bank API (HTTP {response.status_code})."
        rows = [{"year": int(row["date"]), "value": row["value"]} for row in payload[1] if row.get("date") and row.get("value") is not None]
        df = pd.DataFrame(rows, columns=["year", "value"])
        if not df.empty:
            df = df.sort_values("year").reset_index(drop=True)
        return df, None if not df.empty else "The API response contained no non-null observations for this period."
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "The World Bank request timed out. Try again later."
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not retrieve valid World Bank data: {exc}"

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(country, indicator, start, end):
    return fetch_world_bank(country, indicator, start, end)

def validate_data(df, indicator_name):
    errors = []
    if df.empty: return ["There are no observations to validate."]
    if df["year"].isna().any() or not np.all(np.equal(df["year"].dropna(), np.floor(df["year"].dropna()))):
        errors.append("Years must be whole numbers.")
    if df["year"].duplicated().any(): errors.append("Each year must appear only once.")
    if df["value"].isna().any() or not np.isfinite(pd.to_numeric(df["value"], errors="coerce")).all():
        errors.append("Values must be finite numbers; blanks are not allowed.")
    vals = pd.to_numeric(df["value"], errors="coerce")
    if indicator_name in ("Life expectancy at birth", "Unemployment, national estimate", "Crude birth rate", "Intentional homicide rate", "Total reserves including gold") and (vals < 0).any():
        errors.append("This indicator cannot contain negative values.")
    if indicator_name == "Unemployment, national estimate" and (vals > 100).any():
        errors.append("Unemployment percentage cannot exceed 100.")
    return errors

def safe_expr(source, x):
    """Parse a restricted mathematical expression; no Python evaluation."""
    source = source.strip().replace("^", "**")
    if not source or len(source) > 250:
        raise ValueError("Enter a non-empty expression shorter than 250 characters.")
    allowed = {"x": x, "pi": sp.pi, "E": sp.E, "sin": sp.sin, "cos": sp.cos,
               "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
               "abs": sp.Abs}
    try:
        expr = sp.sympify(source, locals=allowed, evaluate=True)
        if not expr.free_symbols.issubset({x}):
            raise ValueError("Use x and supported functions only. Replace a, b, A, etc. with numeric values.")
        if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError("The expression contains an undefined or infinite constant.")
        return expr
    except Exception as exc:
        if isinstance(exc, ValueError): raise
        raise ValueError("Could not parse the function. Use notation such as x**2, 3*x, sin(x), exp(x), or log(x).") from exc

def numeric_eval(expr, x, values):
    fn = sp.lambdify(x, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        out = np.asarray(fn(values), dtype=float)
    if out.ndim == 0: out = np.full(np.asarray(values).shape, float(out))
    return out

def fmt(v, digits=6):
    try:
        if not np.isfinite(float(v)): return "undefined"
        return f"{float(v):,.{digits}g}"
    except Exception: return str(v)

def poly_fit(df, degree):
    base = int(df["year"].min())
    t = df["year"].to_numpy(dtype=float) - base
    y = df["value"].to_numpy(dtype=float)
    if len(np.unique(t)) < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree+1} distinct years; only {len(np.unique(t))} are available.")
    coeff = np.polyfit(t, y, degree)
    pred = np.polyval(coeff, t)
    residual = y - pred
    ss_tot = np.sum((y-y.mean())**2)
    r2 = 1 - np.sum(residual**2)/ss_tot if ss_tot > 0 else float("nan")
    mae = np.mean(np.abs(residual))
    return base, coeff, pred, residual, r2, mae

def poly_expr(coeff, base, x):
    return sp.expand(sum(sp.Float(c, 16)*(x-base)**(len(coeff)-i-1) for i,c in enumerate(coeff)))

def make_plot(observed, model_expr, x, years, values, base, horizon, sample_step, compare=None, point=None, batch=None, title="Historical data and model"):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.scatter(observed["year"], observed["value"], label="Observed World Bank data", color="#1769aa", zorder=4)
    lo, hi = int(observed.year.min()), int(observed.year.max())
    smooth_end = hi + horizon
    xx = np.linspace(lo, max(lo+1, smooth_end), 600)
    yy = numeric_eval(model_expr, x, xx-base)
    hist_mask = xx <= hi
    ax.plot(xx[hist_mask], yy[hist_mask], color="#e87500", lw=2.2, label="Fitted curve (historical range)")
    if horizon > 0:
        ax.plot(xx[~hist_mask], yy[~hist_mask], color="#8e44ad", lw=2, ls="--", label="Extrapolated segment")
    sampled = np.arange(lo, hi+1, sample_step)
    if len(sampled):
        sy = numeric_eval(model_expr, x, sampled-base)
        ax.scatter(sampled, sy, marker="x", s=35, color="#27834a", label=f"Model markers every {sample_step} year(s)")
    if point is not None:
        ax.scatter([point[0]], [point[1]], color="black", marker="D", s=55, label="Single-year estimate", zorder=6)
    if batch is not None and len(batch):
        ax.scatter(batch["year"], batch["estimate"], marker="^", color="#b21f35", label="Batch estimates", zorder=5)
    if compare is not None:
        ax.plot(compare.year, compare.value, marker="o", ms=3, lw=1.4, label="Comparison country (observed)")
    ax.axvline(hi, color="gray", ls=":", lw=1.2, label="Historical boundary")
    ax.set_title(title); ax.set_xlabel("Year"); ax.set_ylabel("Indicator value")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    return fig

def pdf_bytes(title, metadata, equation, metrics, findings, preview, results, chart_bytes):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Paragraph(metadata, styles["BodyText"]), Spacer(1, 8),
             Paragraph("<b>Model equation</b>: " + equation.replace("&","&amp;").replace("<","&lt;"), styles["BodyText"]),
             Paragraph("<b>Metrics</b>: " + metrics, styles["BodyText"]), Spacer(1, 8)]
    if chart_bytes:
        story.append(Image(io.BytesIO(chart_bytes), width=500, height=260))
    story += [Spacer(1, 8), Paragraph("<b>Findings and limitations</b>", styles["Heading2"])]
    for item in findings: story.append(Paragraph("• " + item, styles["BodyText"]))
    story += [Spacer(1, 8), Paragraph("<b>Data preview</b>", styles["Heading2"])]
    if len(preview):
        data = [["Year","Value"]] + [[str(int(r.year)), fmt(r.value,7)] for r in preview.head(12).itertuples()]
        tbl = Table(data, repeatRows=1); tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.3,colors.grey)])); story.append(tbl)
    if results is not None and len(results):
        story += [Spacer(1,8), Paragraph("<b>Calculated year results</b>", styles["Heading2"])]
        data = [["Year","Estimate","Classification"]] + [[str(int(r.year)),fmt(r.estimate,7),r.classification] for r in results.itertuples()]
        tbl = Table(data, repeatRows=1); tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.3,colors.grey)])); story.append(tbl)
    doc.build(story); return buf.getvalue()

# ------------------------------- Interface ---------------------------------
st.title("Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB · historical data, mathematical modeling, and contextual calculus")
st.info("Brazil is included as a Latin American comparison country. The country set is a classroom comparison set; it does not imply that every resident has the same heritage or identity.")

now_year = datetime.now().year
default_start, default_end = now_year-70, now_year-1
with st.sidebar:
    st.header("1 · Historical data")
    country_label = st.selectbox("Featured country", list(COUNTRIES), index=0)
    indicator_name = st.selectbox("World Bank indicator", list(INDICATORS), index=0)
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    st.divider()
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2","Precalculus","Calculus"])
    context = st.selectbox("Application context", list(CONTEXTS))
    st.caption("Historical model and context model are separate: you can analyze the retrieved-data fit or switch to a context function.")
    compare_label = st.selectbox("Comparison country", ["None"] + [c for c in COUNTRIES if c != country_label])
    printer_mode = st.toggle("Printer-friendly report view", value=False)

country_code = COUNTRIES[country_label]
indicator_code, units, definition, caveat = INDICATORS[indicator_name]
source_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={default_start}:{default_end}&format=json"
with st.spinner("Retrieving World Bank observations…"):
    source_df, fetch_error = cached_fetch(country_code, indicator_code, default_start, default_end)
if "source_key" not in st.session_state or st.session_state.source_key != (country_code,indicator_code):
    st.session_state.source_key = (country_code,indicator_code)
    st.session_state.edit_df = source_df.copy()
    st.session_state.calc_single = None
    st.session_state.calc_batch = None
if fetch_error:
    st.warning(fetch_error)
st.caption(f"Source: World Bank · retrieved {date.today().isoformat()} · requested years {default_start}–{default_end}. Actual non-null coverage: {int(source_df.year.min()) if len(source_df) else 'none'}–{int(source_df.year.max()) if len(source_df) else 'none'}; {len(source_df)} observations.")
st.markdown(f"**Definition:** {definition}  \n**Units:** {units}  \n**Interpretation note:** {caveat}  \n[Open World Bank indicator source]({source_url})")

# Editable data and reset
st.subheader("2 · Validate and edit the working data")
if source_df.empty:
    st.error("No source observations are available. The app will not invent substitute values. You can still explore a context function below.")
else:
    with st.expander("Edit year/value observations", expanded=False):
        st.write("This table is a working copy. Original API data remains unchanged.")
        edited = st.data_editor(st.session_state.edit_df, num_rows="dynamic", use_container_width=True, key="data_editor")
        st.session_state.edit_df = edited
        col_a,col_b = st.columns([1,4])
        if col_a.button("Reset to source"):
            st.session_state.edit_df = source_df.copy()
            st.session_state.calc_single = None; st.session_state.calc_batch = None
            st.rerun()
        edited_flag = not st.session_state.edit_df.reset_index(drop=True).equals(source_df.reset_index(drop=True))
        st.caption("Status: USER-MODIFIED SCENARIO" if edited_flag else "Status: original source values")
        data_errors = validate_data(st.session_state.edit_df, indicator_name)
        if data_errors:
            for e in data_errors: st.error(e)
        else:
            st.success(f"Working table valid: {len(st.session_state.edit_df)} observations.")
working = st.session_state.edit_df.copy() if "edit_df" in st.session_state else source_df.copy()
working["year"] = pd.to_numeric(working["year"], errors="coerce")
working["value"] = pd.to_numeric(working["value"], errors="coerce")
errors = validate_data(working, indicator_name)
valid_working = not errors and len(working) > 0
if valid_working:
    working = working.sort_values("year").reset_index(drop=True)

# Fit historical polynomial
st.subheader("3 · Historical polynomial model")
x = sp.Symbol("x", real=True)
fit_available = False
if valid_working:
    try:
        base, coeff, fitted, residuals, r2, mae = poly_fit(working, degree)
        hist_expr = poly_expr(coeff, base, x)
        fit_available = True
        st.caption(f"Centered time: t = year − {base}. The fitted polynomial uses t; the equation below uses x as calendar year.")
        st.latex(sp.latex(sp.expand(hist_expr.subs(x, sp.Symbol("year")))))
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R² · in-sample", fmt(r2,8)); m2.metric("MAE · in-sample", fmt(mae,8))
        m3.metric("Observations", str(len(working))); m4.metric("Historical mean", f"{working.value.mean():,.5g} {units}")
        st.write(f"**Base year:** {base} · **Degree:** {degree} · **Historical range:** {int(working.year.min())}–{int(working.year.max())}")
        residual_df = pd.DataFrame({"year":working.year.astype(int),"actual":working.value,"fitted":fitted,"residual (actual − fitted)":residuals})
        with st.expander("Residuals and model diagnostics"):
            st.dataframe(residual_df, use_container_width=True)
            st.caption("R² and MAE are in-sample fit metrics. They describe fit to the same observations used to estimate the coefficients; they do not prove causation or guarantee future accuracy.")
        # chronological holdout diagnostic
        n = len(working)
        if n >= degree + 4:
            train_n = max(degree+1, int(n*.8))
            train = working.iloc[:train_n]
            test = working.iloc[train_n:]
            try:
                hb,hc,*_ = poly_fit(train, degree)
                holdout_pred = np.polyval(hc, test.year.to_numpy()-hb)
                holdout_mae = float(np.mean(np.abs(test.value.to_numpy()-holdout_pred)))
                st.write(f"**Chronological holdout MAE (separate score):** {fmt(holdout_mae,8)} on {len(test)} later observations.")
            except Exception as exc: st.caption(f"Holdout score unavailable: {exc}")
        else:
            st.caption("Chronological holdout comparison requires at least degree + 4 observations; not enough data are available.")
    except Exception as exc:
        st.error(str(exc))
else:
    hist_expr = None
    if errors:
        for e in errors: st.error(e)
    st.warning("A valid historical fit is unavailable until the working data pass validation and contain enough distinct years.")

# Country comparison same indicator
compare_df = None
if compare_label != "None":
    cc = COUNTRIES[compare_label]
    compare_df, cmp_err = cached_fetch(cc, indicator_code, default_start, default_end)
    if cmp_err: st.warning(f"{compare_label} comparison: {cmp_err}")
    else: st.caption(f"{compare_label} comparison coverage: {int(compare_df.year.min())}–{int(compare_df.year.max())}, {len(compare_df)} observations. Missing years and different coverage are not filled or interpolated.")

# Year estimates
st.subheader("4 · Calculate single or batch estimates")
calc_single_df = None
if fit_available:
    hist_lo, hist_hi = int(working.year.min()), int(working.year.max())
    with st.form("single_year_form"):
        single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=min(2100,max(1950,hist_hi+1)), step=1)
        single_submit = st.form_submit_button("Calculate Value")
    if single_submit:
        estimate = float(numeric_eval(hist_expr, x, np.array([single_year-base]))[0])
        classification = "within fitted historical range" if hist_lo <= single_year <= hist_hi else "outside fitted historical range (extrapolation)"
        st.session_state.calc_single = {"year":int(single_year),"estimate":estimate,"classification":classification,"key":(country_code,indicator_code,degree,tuple(working.year),tuple(working.value))}
    calc_key = (country_code,indicator_code,degree,tuple(working.year),tuple(working.value))
    if st.session_state.get("calc_single") and st.session_state.calc_single.get("key") == calc_key:
        cs = st.session_state.calc_single
        st.metric(f"Estimate for {cs['year']}", f"{cs['estimate']:,.7g} {units}")
        st.caption(f"{cs['classification']}. A fitted value at an observed year is a model estimate, not the actual observed value.")
        calc_single_df = pd.DataFrame([{"year":cs["year"],"estimate":cs["estimate"],"classification":cs["classification"]}])
    elif st.session_state.get("calc_single"):
        st.session_state.calc_single = None
    with st.form("batch_form"):
        batch_mode = st.radio("Batch input", ["Year range","Custom comma-separated years"], horizontal=True)
        if batch_mode == "Year range":
            c1,c2,c3 = st.columns(3)
            y_start = c1.number_input("Start year",1950,2100,value=max(1950,hist_hi+1),step=1)
            y_end = c2.number_input("End year",1950,2100,value=min(2100,max(1950,hist_hi+5)),step=1)
            y_step = c3.number_input("Step",1,10,1,step=1)
            custom_years = ""
        else:
            custom_years = st.text_input("Years separated by commas", "2000, 2010, 2020")
            y_start=y_end=1950; y_step=1
        batch_submit = st.form_submit_button("Calculate batch")
    if batch_submit:
        try:
            if batch_mode == "Year range":
                if y_end < y_start: raise ValueError("End year must be greater than or equal to start year.")
                requested = list(range(int(y_start),int(y_end)+1,int(y_step)))
            else:
                parts = [p.strip() for p in custom_years.split(",") if p.strip()]
                requested = [int(p) for p in parts]
                if not requested: raise ValueError("Enter at least one year.")
            invalid = [v for v in requested if v<1950 or v>2100]
            if invalid: raise ValueError(f"Years must be between 1950 and 2100. Invalid: {invalid}")
            if len(set(requested)) != len(requested): raise ValueError("Remove duplicate requested years.")
            vals = numeric_eval(hist_expr, x, np.array(requested)-base)
            classifications = ["within fitted historical range" if hist_lo<=v<=hist_hi else "outside fitted historical range (extrapolation)" for v in requested]
            st.session_state.calc_batch = pd.DataFrame({"year":requested,"estimate":vals,"classification":classifications})
            st.session_state.batch_key = calc_key
        except Exception as exc: st.error(f"Batch input error: {exc}")
    batch_df = st.session_state.get("calc_batch")
    if batch_df is not None and st.session_state.get("batch_key") == calc_key:
        st.dataframe(batch_df, use_container_width=True)
        st.caption("Batch estimates use the same fitted transformation and calculation as the single-year tool.")
    else:
        batch_df = None
        if st.session_state.get("calc_batch") is not None: st.session_state.calc_batch = None
else:
    batch_df = None

# Context function controls
st.subheader("5 · Function builder and calculus lab")
st.write("Choose a context model or use the polynomial fitted to the retrieved historical data. The context model is a separate mathematical scenario unless you explicitly select the historical fit.")
use_fit = st.checkbox("Use generated function from historical curve of best fit", value=bool(fit_available))
suggestion = SUGGESTIONS[context]
defaults = CONTEXTS[context]
if context == "Engineering":
    default_expr = "0.5*2*x**2 + 3*x + 1"
elif context == "Medical":
    default_expr = "100*exp(-0.25*x) + 2"
elif context in ("Biology","Bioengineering"):
    default_expr = "1000/(1 + 9*exp(-0.4*x))"
elif context == "Weather":
    default_expr = "10*sin(0.5*x) + 15"
elif context == "Sports science":
    default_expr = "4*sin(0.8*x)"
elif context == "Business":
    default_expr = "2*x**2 + 5*x + 20"
elif context == "Economic forecasting":
    default_expr = "0.2*x**2 - 2*x + 100"
else:
    default_expr = "3*x + 4"
with st.expander("Customize context, variables, and function", expanded=True):
    context_description = st.text_area("Context description", defaults[4], height=70)
    c1,c2,c3,c4 = st.columns(4)
    dep_name = c1.text_input("Dependent variable", defaults[0])
    dep_units = c2.text_input("Dependent units", defaults[1])
    indep_name = c3.text_input("Independent variable", defaults[2])
    indep_units = c4.text_input("Independent units", defaults[3])
    function_kind = st.selectbox("Function family", ["Polynomial","Rational","Exponential","Logarithmic","Trigonometric","Custom expression"])
    function_templates = {
        "Polynomial": "2*x**2 - 3*x + 1", "Rational": "(x**2 + 1)/(x - 2)",
        "Exponential": "3*exp(0.4*x) + 2", "Logarithmic": "4*log(x) + 1",
        "Trigonometric": "5*sin(0.5*x) + 10", "Custom expression": default_expr
    }
    function_text = st.text_input("Function f(x)", function_templates[function_kind] if function_kind!="Custom expression" else default_expr)
    st.caption(f"Suggested for {context}: `{suggestion}`. Replace letters with numbers; supported functions: sin, cos, tan, exp, log, sqrt, abs, pi, E.")
    a,b = st.columns(2)
    default_analysis_start = float(working.year.min()) if use_fit and fit_available else 0.0
    default_analysis_end = float(working.year.max()) if use_fit and fit_available else 10.0
    analysis_start = a.number_input("Analysis interval start", value=default_analysis_start)
    analysis_end = b.number_input("Analysis interval end", value=default_analysis_end)
    if analysis_end <= analysis_start: st.error("Analysis interval end must be greater than its start.")
    sim_visual = st.selectbox("Contextual visual simulation", ["Walking person with prosthetic leg","Moving object / projectile","Tank or concentration indicator","Population / cell colony","Thermometer / weather indicator","Business / economic indicator"])
    st.caption("Simulation drawings are schematic. Their scale is illustrative and is not a medical device, diagnosis, or physical prediction.")
    st.session_state.sim_speed = st.slider("Playback speed",1,10,4)
    if "sim_x" not in st.session_state: st.session_state.sim_x = float(analysis_start)
    sim_x = st.slider(f"Current {indep_name} ({indep_units})", float(analysis_start), float(analysis_end), float(min(max(st.session_state.sim_x,analysis_start),analysis_end)), step=max((analysis_end-analysis_start)/200,0.001))
    st.session_state.sim_x = sim_x
    bplay,bpause,breset = st.columns(3)
    if bplay.button("▶ Play"):
        st.session_state.playing = True
    if bpause.button("Ⅱ Pause"):
        st.session_state.playing = False
    if breset.button("↺ Reset"):
        st.session_state.playing = False; st.session_state.sim_x = float(analysis_start); st.rerun()

context_expr = None
context_error = None
if analysis_end > analysis_start:
    try:
        if use_fit:
            if not fit_available: raise ValueError("Historical fit is unavailable; choose a context function or repair the working data.")
            # Convert centered historical t to context variable x while retaining the same fit.
            context_expr = poly_expr(coeff, base, x)
            function_text_display = f"Historical degree-{degree} polynomial in t = year − {base}"
        else:
            context_expr = safe_expr(function_text, x)
            function_text_display = function_text
    except Exception as exc:
        context_error = str(exc); st.error(f"Function error: {context_error}")
if context_expr is not None:
    d1 = sp.diff(context_expr,x); d2 = sp.diff(d1,x)
    st.markdown(f"**Function:** {dep_name} = f({indep_name})")
    st.latex("f(x) = " + sp.latex(context_expr))
    st.markdown("**First derivative**"); st.latex("f'(x) = " + sp.latex(d1))
    st.markdown("**Second derivative**"); st.latex("f''(x) = " + sp.latex(d2))
    st.caption(f"Units: f(x) in {dep_units}; f′(x) in {dep_units}/{indep_units}; f″(x) in {dep_units}/{indep_units}².")
    xx = np.linspace(float(analysis_start),float(analysis_end),900)
    yy = numeric_eval(context_expr,x,xx)
    dy = numeric_eval(d1,x,xx); ddy = numeric_eval(d2,x,xx)
    valid = np.isfinite(yy)
    # Plot gaps rather than connecting across undefined values.
    show_f = st.checkbox("Show function",True,key="show_f")
    show_d1 = st.checkbox("Show first derivative",False,key="show_d1")
    show_d2 = st.checkbox("Show second derivative",False,key="show_d2")
    fig2,ax2 = plt.subplots(figsize=(10,4.8))
    if show_f: ax2.plot(xx,np.where(valid,yy,np.nan),label="f(x)")
    if show_d1: ax2.plot(xx,np.where(np.isfinite(dy),dy,np.nan),label="f′(x)")
    if show_d2: ax2.plot(xx,np.where(np.isfinite(ddy),ddy,np.nan),label="f″(x)")
    ax2.axvline(sim_x,color="black",ls=":",label="Current point")
    ax2.grid(alpha=.25); ax2.set_xlabel(f"{indep_name} ({indep_units})"); ax2.set_ylabel("Function / derivative values"); ax2.legend(); fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    try:
        fval=float(numeric_eval(context_expr,x,np.array([sim_x]))[0])
        dval=float(numeric_eval(d1,x,np.array([sim_x]))[0])
        ddval=float(numeric_eval(d2,x,np.array([sim_x]))[0])
        st.markdown(f"**At {indep_name} = {fmt(sim_x)} {indep_units}:** {dep_name} = {fmt(fval)} {dep_units}; rate = {fmt(dval)} {dep_units}/{indep_units}; changing rate = {fmt(ddval)} {dep_units}/{indep_units}².")
        if context == "Engineering":
            st.write(f"Interpretation: position is {fmt(fval)} m, velocity is {fmt(dval)} m/s, and acceleration is {fmt(ddval)} m/s².")
        elif context in ("Business","Economic forecasting"):
            st.write(f"Interpretation: the modeled {dep_name.lower()} is {fmt(fval)} {dep_units}; it is changing at {fmt(dval)} {dep_units} per {indep_units}. The second derivative describes whether that rate is increasing or decreasing.")
        else:
            st.write(f"Interpretation: the modeled {dep_name.lower()} is {fmt(fval)} {dep_units}; its instantaneous rate is {fmt(dval)} {dep_units} per {indep_units}, and its rate is changing by {fmt(ddval)} {dep_units} per {indep_units} squared.")
    except Exception:
        st.warning("The selected function is undefined at the current point. Move the point to a valid part of the domain.")

    # Average rate and secant
    st.markdown("**Average rate of change and secant line**")
    r1,r2 = st.columns(2)
    xa = r1.slider("Interval start a",float(analysis_start),float(analysis_end),float(analysis_start),key="sec_a")
    xb = r2.slider("Interval end b",float(analysis_start),float(analysis_end),float(analysis_end),key="sec_b")
    if xb != xa:
        fa,fb = numeric_eval(context_expr,x,np.array([xa,xb]))
        if np.isfinite(fa).all() and np.isfinite(fb).all():
            avg=(fb-fa)/(xb-xa)
            st.write(f"Average rate = (f(b) − f(a))/(b − a) = **{fmt(avg)} {dep_units}/{indep_units}**.")
            secant = fa + avg*(xx-xa)
            fig3,ax3=plt.subplots(figsize=(9,3.2)); ax3.plot(xx,np.where(valid,yy,np.nan),label="f(x)"); ax3.plot(xx,secant,ls="--",label="Secant line"); ax3.scatter([xa,xb],[fa,fb]); ax3.set_ylim(*np.nanpercentile(np.r_[yy,secant],[2,98])); ax3.grid(alpha=.2); ax3.legend(); st.pyplot(fig3,use_container_width=True)
        else: st.warning("The function is undefined at one or both interval endpoints.")
    else: st.warning("Choose two different interval endpoints.")
    st.markdown("**Instantaneous rate and tangent line**")
    if np.isfinite(dval) and np.isfinite(fval):
        tangent=fval+dval*(xx-sim_x)
        fig4,ax4=plt.subplots(figsize=(9,3.2)); ax4.plot(xx,np.where(valid,yy,np.nan),label="f(x)"); ax4.plot(xx,tangent,ls="--",label="Tangent line"); ax4.scatter([sim_x],[fval]); ax4.grid(alpha=.2); ax4.legend(); st.pyplot(fig4,use_container_width=True)
    # Numerical critical/inflection point analysis
    st.markdown("**Calculus analysis**" if course=="Calculus" else "**Function behavior**")
    try:
        roots1 = [complex(r) for r in sp.nroots(sp.together(d1).as_numer_denom()[0]) if abs(complex(r).imag)<1e-8]
        roots2 = [complex(r) for r in sp.nroots(sp.together(d2).as_numer_denom()[0]) if abs(complex(r).imag)<1e-8]
    except Exception:
        roots1=[]; roots2=[]
    def inside_unique(roots):
        return sorted(set(round(float(r.real),7) for r in roots if analysis_start <= r.real <= analysis_end))
    crit = inside_unique(roots1); inflect_candidates=inside_unique(roots2)
    def sign_at(expr,z):
        val=float(numeric_eval(expr,x,np.array([z]))[0])
        return np.sign(val) if np.isfinite(val) else np.nan
    maxima=[]; minima=[]; inc=[]; dec=[]
    for z in crit:
        eps=max((analysis_end-analysis_start)*1e-5,1e-6)
        sl=sign_at(d1,z-eps); sr=sign_at(d1,z+eps)
        if sl>0 and sr<0: maxima.append(z)
        elif sl<0 and sr>0: minima.append(z)
    cuts=[analysis_start]+crit+[analysis_end]
    for l,r in zip(cuts[:-1],cuts[1:]):
        s=sign_at(d1,(l+r)/2)
        if s>0: inc.append((l,r))
        elif s<0: dec.append((l,r))
    inflections=[]
    for z in inflect_candidates:
        eps=max((analysis_end-analysis_start)*1e-5,1e-6)
        sl=sign_at(d2,z-eps); sr=sign_at(d2,z+eps)
        if np.isfinite(sl) and np.isfinite(sr) and sl*sr<0: inflections.append(z)
    def intervals_text(items): return ", ".join(f"({fmt(a)}, {fmt(b)})" for a,b in items) or "none identified"
    st.write(f"**Increasing:** {intervals_text(inc)} · **Decreasing:** {intervals_text(dec)}")
    st.write(f"**Local maxima (first-derivative sign change):** {', '.join(fmt(z) for z in maxima) or 'none identified'}")
    st.write(f"**Local minima (first-derivative sign change):** {', '.join(fmt(z) for z in minima) or 'none identified'}")
    st.write(f"**Inflection points (second-derivative sign change):** {', '.join(fmt(z) for z in inflections) or 'none confirmed'}")
    # closed interval extrema
    try:
        candidates=[float(analysis_start),float(analysis_end)]+crit
        cvals=numeric_eval(context_expr,x,np.array(candidates))
        finite_pairs=[(z,v) for z,v in zip(candidates,cvals) if np.isfinite(v)]
        if finite_pairs:
            absolute_min=min(finite_pairs,key=lambda q:q[1]); absolute_max=max(finite_pairs,key=lambda q:q[1])
            st.write(f"**Absolute minimum on closed interval (candidate check):** x={fmt(absolute_min[0])}, f(x)={fmt(absolute_min[1])}.")
            st.write(f"**Absolute maximum on closed interval (candidate check):** x={fmt(absolute_max[0])}, f(x)={fmt(absolute_max[1])}.")
    except Exception: pass
    if course=="Honors Algebra 2":
        st.write("**Algebra 2 focus:** The domain is restricted to inputs where the expression is defined. The displayed graph shows the range over the selected interval. End behavior depends on the leading terms for polynomials; average rates compare changes over intervals.")
    elif course=="Precalculus":
        st.write("**Precalculus focus:** Degree and coefficients affect polynomial shape, turning points, and end behavior. Concavity is indicated by whether average rates are increasing or decreasing; derivative-based results are numerical approximations where appropriate.")
    else:
        st.write("**Calculus focus:** Critical points are not automatically extrema; the first-derivative sign change classifies local extrema. A zero of the second derivative is only an inflection candidate; concavity must change. Root and sign tests are numerical approximations.")
    # domain/discontinuity warning
    if context_expr.is_rational_function:
        st.warning("Rational function: denominator zeros are excluded from the domain. Graph gaps are retained at undefined points.")
    if context_expr.has(sp.log):
        st.warning("Logarithmic function: its real-valued domain requires positive logarithm arguments.")
    if context_expr.has(sp.tan):
        st.warning("Tangent function has periodic vertical asymptotes; graph samples near undefined points are omitted.")
    if context_expr.is_polynomial(x):
        st.caption(f"Polynomial end behavior is governed by its leading term: {sp.LC(sp.Poly(context_expr,x))}·x^{sp.degree(context_expr,x)}.")
    # schematic responsive simulation
    st.markdown("**Contextual visual simulation**")
    progress = (sim_x-analysis_start)/(analysis_end-analysis_start)
    if sim_visual=="Walking person with prosthetic leg":
        st.write("🚶  Schematic gait position: " + "▰"*int(progress*20) + "○" + "▱"*max(0,19-int(progress*20)))
        st.caption("The marker follows the selected input value; it illustrates a changing model value, not clinical gait data.")
    elif sim_visual=="Moving object / projectile":
        st.write("🎯 Object position in the selected simulation interval:")
        st.progress(float(progress))
    elif sim_visual=="Tank or concentration indicator":
        st.write("🧪 Modeled concentration / tank indicator:")
        st.progress(float(np.clip(progress,0,1)))
    elif sim_visual=="Population / cell colony":
        st.write("🦠 Population / cell-colony timeline:")
        st.progress(float(progress))
    elif sim_visual=="Thermometer / weather indicator":
        st.write(f"🌡️ Current modeled reading: {fmt(fval)} {dep_units}")
        st.progress(float(progress))
    else:
        st.write(f"📈 Economic / business timeline: {fmt(fval)} {dep_units}")
        st.progress(float(progress))
    if st.session_state.get("playing",False):
        st.caption("Playback is enabled as a classroom state indicator. Move the current-point slider to advance; pause or reset with the controls above. Automatic timed playback is not enabled in this deployment-safe version.")

# Historical graph
st.subheader("6 · Historical visualization")
historical_fig = None
if fit_available:
    historical_fig = make_plot(working,hist_expr,x,working.year.to_numpy(),working.value.to_numpy(),base,horizon,sample_step,compare_df,tuple(calc_single_df.iloc[0][["year","estimate"]]) if calc_single_df is not None else None,batch_df)
    st.pyplot(historical_fig,use_container_width=True)
    st.caption("The smooth fitted curve is separate from the sample markers. The dashed segment is extrapolation, not observed history. Polynomial extrapolations can become implausible quickly; values are not clipped.")
else:
    if len(source_df):
        fig0,ax0=plt.subplots(figsize=(9,4)); ax0.scatter(source_df.year,source_df.value); ax0.set_xlabel("Year"); ax0.set_ylabel(units); ax0.grid(alpha=.2); st.pyplot(fig0,use_container_width=True)

# Export current model and calculations
st.subheader("7 · Export report and data")
edited_flag = valid_working and not working.reset_index(drop=True).equals(source_df.reset_index(drop=True))
metadata = f"{country_label} · {indicator_name} ({indicator_code}) · Retrieved {date.today().isoformat()} · Working data: {'user-modified scenario' if edited_flag else 'source data'}"
if fit_available:
    equation_str = str(sp.expand(hist_expr)).replace("**","^")
    metrics_str = f"R² in-sample={fmt(r2,8)}; MAE in-sample={fmt(mae,8)}; n={len(working)}; base year={base}; degree={degree}"
    findings = [definition,caveat,"Polynomial regression is descriptive, not causal. Extrapolation outside observed years is uncertain and may be implausible.","No validated 95% prediction interval is supplied. Residuals and MAE are descriptive measures of in-sample error."]
    if len(working): findings.append(f"Observed values range from {fmt(working.value.min())} to {fmt(working.value.max())} {units}.")
    preview = working.copy()
    results_export = batch_df if batch_df is not None else calc_single_df
    csv_data = working.to_csv(index=False).encode("utf-8")
    st.download_button("Download working data CSV",csv_data,file_name="hispanic_heritage_working_data.csv",mime="text/csv")
    excel_buf=io.BytesIO()
    with pd.ExcelWriter(excel_buf,engine="xlsxwriter") as writer:
        pd.DataFrame({"Item":["Country","Indicator","Indicator code","Units","Definition","Retrieved","Data status","Degree","Base year","R-squared (in-sample)","MAE (in-sample)","Equation"],"Value":[country_label,indicator_name,indicator_code,units,definition,date.today().isoformat(),"User-modified scenario" if edited_flag else "Source data",degree,base,r2,mae,equation_str]}).to_excel(writer,sheet_name="Summary",index=False)
        working.to_excel(writer,sheet_name="Working data",index=False)
        residual_df.to_excel(writer,sheet_name="Model diagnostics",index=False)
        if results_export is not None: results_export.to_excel(writer,sheet_name="Calculated years",index=False)
    st.download_button("Download Excel workbook (.xlsx)",excel_buf.getvalue(),file_name="hispanic_heritage_data_lab.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    chart_buf=io.BytesIO(); historical_fig.savefig(chart_buf,format="png",dpi=150,bbox_inches="tight")
    pdf_data=pdf_bytes("Hispanic Heritage Historical Data Explorer",metadata,equation_str,metrics_str,findings,preview,results_export,chart_buf.getvalue())
    st.download_button("Download PDF report",pdf_data,file_name="hispanic_heritage_report.pdf",mime="application/pdf")
else:
    st.info("Exports that require a fitted historical model become available after valid data and a successful fit are available.")
    if len(working):
        st.download_button("Download working data CSV",working.to_csv(index=False).encode(),file_name="working_data.csv",mime="text/csv")

with st.expander("How the app works, assumptions, and limitations"):
    st.markdown("""
    **Data flow:** the app requests annual indicator observations from the World Bank API, keeps the original response separate from the editable working table, validates the table, then fits a polynomial to the working observations. It uses centered time `t = year − base year` to reduce numerical scaling problems. Predictions reuse that same centered transformation.

    **Mathematics:** a degree-*d* polynomial requires at least *d* + 1 distinct years. R² and MAE are in-sample metrics. When enough observations exist, a chronological holdout MAE is reported separately. The model is descriptive and does not establish why a historical change occurred.

    **Uncertainty:** residuals and MAE describe observed fitting error; they are not 95% prediction intervals. Polynomial extrapolation can be unstable and is flagged as outside the fitted range. No predictions are silently clipped.

    **Calculus:** symbolic derivatives are calculated with SymPy. Root-finding and sign tests are numerical; exact classifications may not be possible for every expression. Critical points are classified only when a first-derivative sign change is detected. Inflection candidates are confirmed only when sampled concavity changes. Rational/logarithmic/trigonometric expressions can have restricted domains and discontinuities; plotted non-finite samples are omitted.

    **API and source:** World Bank data can be revised, missing for some years, or reported with country-specific coverage differences. Retrieval date, actual coverage, units, definitions, and indicator codes are displayed. An API failure never triggers fabricated replacement values.

    **Simulation:** the visual indicators are schematic classroom aids, not physical or clinical simulations. Automatic timed playback is not included; the selected point, graph marker, tangent, values, and explanations are synchronized through the current-point control.
    """)
st.caption("Project files: app.py · requirements.txt · README.md · sources.md")
