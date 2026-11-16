SELECT
    id_subsistema,
    data_referencia,
    val_cargaenergiamwmed
FROM {{ ref('silver_carga_energia') }}
WHERE val_cargaenergiamwmed <= 0
