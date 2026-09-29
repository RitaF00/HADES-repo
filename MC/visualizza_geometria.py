import dbetto
import pygeomhades
import pygeomtools


# Percorso dei metadati HADES
metadata_path = "/global/u2/r/ritaferi/HADES_DATA/hades-metadata"

print("Caricamento dei metadati...")
db = dbetto.TextDB(metadata_path)


# Configurazione da visualizzare
configuration = (
    db.hardware.configuration
    .V02162B
    .c1
    .th_HS2_lat_psa
    .run0001
)

print("Costruzione della geometria...")
reg = pygeomhades.core.construct(configuration)


print("Apertura del visualizzatore 3D...")
pygeomtools.viewer.visualize(reg)

print("Visualizzatore chiuso.")
