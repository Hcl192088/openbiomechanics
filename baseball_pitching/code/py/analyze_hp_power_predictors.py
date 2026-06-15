import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score


DATA_PATH = "D:/baseball/pitching/obp/high_performance/data/hp_obp.csv"

COLS = {
    "y": "pitch_speed_mph",
    "cmj_peak_power": "peak_power_[w]_mean_cmj",
    "cmj_peak_power_bm": "peak_power_/_bm_[w/kg]_mean_cmj",
    "cmj_p1_asym": "p1_concentric_impulse_asymmetry_[%_l,r]_mean_cmj",
    "cmj_p2_asym": "p2_concentric_impulse_asymmetry_[%_l,r]_mean_cmj",
    "pp_takeoff_force": "peak_takeoff_force_[n]_mean_pp",
    "pp_ecc_force": "peak_eccentric_force_[n]_mean_pp",
}

PLYO_POWER_TERMS = ("plyo", "pushup", "pp")
POWER_TERMS = ("power",)


def find_plyo_power_columns(data):
    return [
        column
        for column in data.columns
        if any(term in column.lower() for term in PLYO_POWER_TERMS)
        and any(term in column.lower() for term in POWER_TERMS)
    ]


def fit_model(data, predictors):
    columns = [COLS["y"]] + [COLS[predictor] for predictor in predictors]
    subset = data[columns].dropna()
    y = subset[COLS["y"]].to_numpy()
    x = subset[[COLS[predictor] for predictor in predictors]].to_numpy()

    model = LinearRegression().fit(x, y)
    r2 = r2_score(y, model.predict(x))
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - x.shape[1] - 1)

    cv_r2 = float("nan")
    if len(y) >= 20:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(LinearRegression(), x, y, cv=cv, scoring="r2")
        cv_r2 = scores.mean()

    return {
        "n": len(subset),
        "r2": round(r2, 4),
        "adj_r2": round(adj_r2, 4),
        "cv_r2_5fold": round(cv_r2, 4),
        "coef": {
            predictor: round(float(coef), 6)
            for predictor, coef in zip(predictors, model.coef_)
        },
    }


def corr(data, predictor):
    subset = data[[COLS["y"], COLS[predictor]]].dropna()
    r_value, p_value = stats.pearsonr(subset[COLS[predictor]], subset[COLS["y"]])
    return {
        "n": len(subset),
        "r": round(float(r_value), 4),
        "r2": round(float(r_value * r_value), 4),
        "p": float(p_value),
    }


def print_model(label, data, predictors):
    print(label, predictors, fit_model(data, predictors))


def main():
    data = pd.read_csv(DATA_PATH)

    print("rows", len(data))
    for label, column in COLS.items():
        print(label, column, "non-null", int(data[column].notna().sum()))

    plyo_power_columns = find_plyo_power_columns(data)
    print("\nPLYO PUSHUP POWER COLUMN CHECK")
    if plyo_power_columns:
        for column in plyo_power_columns:
            print(column)
    else:
        print("No plyo pushup peak-power column found in hp_obp.csv.")

    print("\nUNIVARIATE")
    for predictor in [
        "cmj_peak_power",
        "cmj_peak_power_bm",
        "cmj_p1_asym",
        "cmj_p2_asym",
        "pp_takeoff_force",
        "pp_ecc_force",
    ]:
        print(predictor, corr(data, predictor))

    print("\nCOMPLETE CASE: CMJ peak vs CMJ P1/P2")
    cmj_case = data[
        [
            COLS["y"],
            COLS["cmj_peak_power"],
            COLS["cmj_p1_asym"],
            COLS["cmj_p2_asym"],
        ]
    ].dropna()
    for predictors in [
        ["cmj_peak_power"],
        ["cmj_p1_asym"],
        ["cmj_p2_asym"],
        ["cmj_p1_asym", "cmj_p2_asym"],
        ["cmj_peak_power", "cmj_p1_asym", "cmj_p2_asym"],
    ]:
        print_model("cmj", cmj_case, predictors)

    cmj_base = fit_model(cmj_case, ["cmj_peak_power"])["r2"]
    cmj_full = fit_model(
        cmj_case, ["cmj_peak_power", "cmj_p1_asym", "cmj_p2_asym"]
    )["r2"]
    cmj_asym = fit_model(cmj_case, ["cmj_p1_asym", "cmj_p2_asym"])["r2"]
    print("delta asym over cmj_peak_power", round(cmj_full - cmj_base, 4))
    print("delta cmj_peak_power over p1p2", round(cmj_full - cmj_asym, 4))

    print("\nCOMPLETE CASE: Plyo pushup force vs CMJ peak power")
    pp_case = data[
        [
            COLS["y"],
            COLS["cmj_peak_power"],
            COLS["cmj_peak_power_bm"],
            COLS["pp_takeoff_force"],
            COLS["pp_ecc_force"],
        ]
    ].dropna()
    for predictors in [
        ["cmj_peak_power"],
        ["cmj_peak_power_bm"],
        ["pp_takeoff_force"],
        ["pp_ecc_force"],
        ["pp_takeoff_force", "pp_ecc_force"],
        ["cmj_peak_power", "pp_takeoff_force"],
        ["cmj_peak_power", "pp_takeoff_force", "pp_ecc_force"],
    ]:
        print_model("pp", pp_case, predictors)

    pp_base = fit_model(pp_case, ["cmj_peak_power"])["r2"]
    pp_full = fit_model(
        pp_case, ["cmj_peak_power", "pp_takeoff_force", "pp_ecc_force"]
    )["r2"]
    pp_only = fit_model(pp_case, ["pp_takeoff_force", "pp_ecc_force"])["r2"]
    print("delta pp forces over cmj_peak_power", round(pp_full - pp_base, 4))
    print("delta cmj_peak_power over pp forces", round(pp_full - pp_only, 4))

    print("\nPREDICTOR CORR COMPLETE PP CASE")
    print(
        pp_case[
            [
                COLS["cmj_peak_power"],
                COLS["pp_takeoff_force"],
                COLS["pp_ecc_force"],
            ]
        ]
        .corr()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
