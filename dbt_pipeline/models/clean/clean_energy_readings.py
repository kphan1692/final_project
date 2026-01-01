import pandas as pd

#This function is a dbt model that loads a staging table, 
# cleans time and numeric columns, fills gaps by device, 
# and returns a table materialized in the "clean" schema.
def model(dbt, session):
    dbt.config(materialized="table", schema="clean")

    df = dbt.ref("stg_energy_readings").df()

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["id_device", "ts"])

    target = "ac_power"

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id_device"]

    def fill_group(g):
        g = g.set_index("ts")
        g[numeric_cols] = g[numeric_cols].interpolate(method="time").ffill().bfill()
        return g.reset_index()

    df = df.groupby("id_device", group_keys=False).apply(fill_group)
    df = df.dropna(subset=[target])

    return df
