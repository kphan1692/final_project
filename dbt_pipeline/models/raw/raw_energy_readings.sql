{{ config(materialized='table', schema='raw') }}

select *
from read_csv_auto('{{ var("raw_csv_path") }}', header=true)
