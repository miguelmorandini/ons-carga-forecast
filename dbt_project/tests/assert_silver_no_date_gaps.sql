WITH date_diffs AS (
    SELECT
        id_subsistema,
        data_referencia,
        LAG(data_referencia) OVER (
            PARTITION BY id_subsistema
            ORDER BY data_referencia
        ) AS prev_data_referencia
    FROM {{ ref('silver_carga_energia') }}
)

SELECT
    id_subsistema,
    prev_data_referencia,
    data_referencia,
    data_referencia - prev_data_referencia AS gap_days
FROM date_diffs
WHERE prev_data_referencia IS NOT NULL
  AND (data_referencia - prev_data_referencia) > 1
