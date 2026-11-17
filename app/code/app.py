import os
import pickle

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State

# ---------- Load the saved model (preprocessing + Gradient Boosting) ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "car_price_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# The dropdown options come from the categories the model saw in training
onehot = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
fuels, transmissions, brands = [list(c) for c in onehot.categories_]

FEATURES = ["year", "max_power", "engine", "km_driven", "owner", "fuel", "transmission", "brand"]


def predict_price(values):
    """Predict the price of one car. Empty fields become NaN, and the
    pipeline fills them in (median for numbers, most common for categories)."""
    row = {f: (np.nan if values.get(f) in (None, "") else values[f]) for f in FEATURES}
    log_price = model.predict(pd.DataFrame([row]))[0]
    return float(np.exp(log_price))  # the model predicts log(price), so undo the log


# ---------- Page layout ----------
app = Dash(__name__)


def field(label, component):
    return html.Div([html.Label(label, style={"fontWeight": "bold"}), component],
                    style={"marginBottom": "12px"})


app.layout = html.Div(style={"maxWidth": "600px", "margin": "40px auto", "fontFamily": "sans-serif"}, children=[
    html.H1("Car Price Prediction"),
    html.P("Enter the details of the car and click Predict. "
           "If you don't know a value, leave it empty: the model fills it in "
           "with a typical value from the training data (the median for numbers, "
           "the most common option for categories)."),

    field("Brand", dcc.Dropdown(id="brand", options=brands, placeholder="e.g. Maruti")),
    field("Year of manufacture", dcc.Input(id="year", type="number", placeholder="e.g. 2017")),
    field("Max power (bhp)", dcc.Input(id="max_power", type="number", placeholder="e.g. 82")),
    field("Engine size (CC)", dcc.Input(id="engine", type="number", placeholder="e.g. 1197")),
    field("Kilometres driven", dcc.Input(id="km_driven", type="number", placeholder="e.g. 40000")),
    field("Number of previous owners", dcc.Dropdown(id="owner", options=[
        {"label": "First owner", "value": 1}, {"label": "Second owner", "value": 2},
        {"label": "Third owner", "value": 3}, {"label": "Fourth & above", "value": 4}])),
    field("Fuel", dcc.Dropdown(id="fuel", options=fuels)),
    field("Transmission", dcc.Dropdown(id="transmission", options=transmissions)),

    html.Button("Predict", id="predict", n_clicks=0),
    html.H2(id="result", style={"marginTop": "20px"}),
])


# ---------- What happens when the button is clicked ----------
@app.callback(
    Output("result", "children"),
    Input("predict", "n_clicks"),
    [State(f, "value") for f in FEATURES],
    prevent_initial_call=True,
)
def on_predict(n_clicks, *values):
    price = predict_price(dict(zip(FEATURES, values)))
    return f"Predicted selling price: {price:,.0f}"


if __name__ == "__main__":
    # host 0.0.0.0 so the app is reachable from outside the Docker container
    app.run(host="0.0.0.0", port=8050, debug=False)
