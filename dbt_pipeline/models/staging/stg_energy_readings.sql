{{ config(materialized='view', schema='staging') }}

with src as (
  select * from {{ ref('raw_energy_readings') }}
),
parsed as (
  select
    *,
    coalesce(
      try_strptime(cast(time as varchar), '%d/%m/%Y %H:%M:%S'),
      try_strptime(cast(time as varchar), '%m/%d/%Y %H:%M:%S'),
      try_strptime(cast(time as varchar), '%Y-%m-%d %H:%M:%S'),
      try_strptime(cast(time as varchar), '%Y-%m-%d %H:%M:%S.%f'),
      try_strptime(cast(time as varchar), '%Y-%m-%dT%H:%M:%S'),
      try_strptime(cast(time as varchar), '%Y-%m-%dT%H:%M:%S.%f')
    ) as ts
  from src
)
select
  ts,
  cast(id_device as bigint) as id_device,

  cast(ac_power as double) as ac_power,
  cast(nvmActivePower as double) as nvm_active_power,
  cast(last_15min_kwh as double) as last_15min_kwh,

  parsed.* exclude (ts)
from parsed
where ts is not null
