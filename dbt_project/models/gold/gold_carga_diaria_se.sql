SELECT
    data_referencia,
    id_subsistema,
    nom_subsistema,
    val_cargaenergiamwmed
FROM {{ ref('silver_carga_energia') }}
WHERE id_subsistema = 'SE'
ORDER BY data_referencia
