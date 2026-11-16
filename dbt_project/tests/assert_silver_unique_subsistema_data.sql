SELECT
    id_subsistema,
    data_referencia,
    COUNT(*) AS total_registros
FROM {{ ref('silver_carga_energia') }}
GROUP BY id_subsistema, data_referencia
HAVING COUNT(*) > 1
