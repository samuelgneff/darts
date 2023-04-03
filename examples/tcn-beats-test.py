import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models.forecasting.nbeats_tcn import NBEATSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score
from darts.datasets import EnergyDataset

df = EnergyDataset().load().pd_dataframe()
df_day_avg = df.groupby(df.index.astype(str).str.split(" ").str[0]).mean().reset_index()
filler = MissingValuesFiller()
scaler = Scaler()
series = scaler.fit_transform(
    filler.transform(
        TimeSeries.from_dataframe(
            df_day_avg, "time", ["generation hydro run-of-river and poundage"]
        )
    )
).astype(np.float32)
train, val = series.split_after(pd.Timestamp("20170901"))
model_nbeats = NBEATSModel(
    input_chunk_length=13,
    output_chunk_length=7,
    generic_architecture=True,
    num_stacks=2,
    num_blocks=1,
    num_layers=3,
    layer_widths=13,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=60,
    model_name="nbeats_run",
    kernel_size= 4,
    dilation_base=2,
    weight_norm=True,
    pl_trainer_kwargs={'accelerator': 'cpu'}
)

model_nbeats.fit(train, val_series=val, verbose=True)
