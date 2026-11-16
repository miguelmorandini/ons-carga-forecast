WITH raw_carga AS (
    SELECT
        TRIM(CAST(id_subsistema AS VARCHAR)) AS id_subsistema,
        TRIM(CAST(nom_subsistema AS VARCHAR)) AS nom_subsistema,
        CAST(din_instante AS DATE) AS data_referencia,
        CAST(val_cargaenergiamwmed AS DOUBLE) AS val_cargaenergiamwmed,
        din_instante
    FROM read_parquet('data/bronze/CARGA_ENERGIA_*.parquet')
    WHERE din_instante IS NOT NULL
      AND val_cargaenergiamwmed IS NOT NULL
)

SELECT
    id_subsistema,
    nom_subsistema,
    data_referencia,
    val_cargaenergiamwmed
FROM raw_carga
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY id_subsistema, data_referencia
    ORDER BY din_instante DESC
) = 1
