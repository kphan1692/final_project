{{ config(materialized='table', schema='mart') }}

select
  id_device,
  ts,
  ac_power as target_ac_power,
  nvm_active_power,
  last_15min_kwh,

  extract('year' from ts) as year,
  extract('month' from ts) as month,
  extract('day' from ts) as day,
  extract('hour' from ts) as hour,
  extract('minute' from ts) as minute
from {{ ref('clean_energy_readings') }}
